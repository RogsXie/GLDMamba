import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, trunc_normal_, to_2tuple
from einops import rearrange

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.wq = nn.Linear(input_dim, input_dim)
        self.wk = nn.Linear(input_dim, input_dim)
        self.wv = nn.Linear(input_dim, input_dim)
        # self.scale = torch.sqrt(torch.tensor(input_dim, dtype=torch.float32))

        self.scale = (float(input_dim)) ** 0.5

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, h*w, c)
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)

        energy = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attention = F.softmax(energy, dim=-1)

        x = torch.matmul(attention, value)
        x = x.view(b, c, h, w)
        return x


class EntropyWeighting(nn.Module):
    def __init__(self):
        super(EntropyWeighting, self).__init__()
        # ✅ 先注册一个空 buffer；以后赋值会保持为 buffer
        self.register_buffer("positional_encoding", torch.empty(0), persistent=False)

    def forward(self, x):
        b, c, h, w = x.size()

        # ✅ 如果第一次使用、形状/设备/精度不匹配，就用输入的 device/dtype 重新分配
        if (self.positional_encoding.numel() == 0 or
            self.positional_encoding.shape != (1, c, h, w) or
            self.positional_encoding.device != x.device or
            self.positional_encoding.dtype  != x.dtype):
            self.positional_encoding = torch.zeros(1, c, h, w, device=x.device, dtype=x.dtype)

        # 现在两者都在同一设备了
        x = x + self.positional_encoding

        x_flat = x.view(b, c, h * w)
        x_exp = torch.exp(x_flat)
        data_sum = torch.sum(x_exp, dim=2, keepdim=True)
        p = x_exp / (data_sum + 1e-9)
        e = -torch.sum(p * torch.log(p + 1e-9), dim=1)
        e_min = torch.min(e)
        e_max = torch.max(e)
        e_normalized = (e - e_min) / (e_max - e_min + 1e-9)
        d = 1 - e_normalized
        weights = d / torch.sum(d, dim=1, keepdim=True)
        R = weights.unsqueeze(1) * x_flat
        output = R.view(b, c, h, w)
        return output


class PatchEmbed(nn.Module):

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=5, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))

        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))

    def forward(self, x1, x2, H, W):
        B, N, C = x1.shape
        v = self.q(x2)

        if self.sr_ratio > 1:
            x_ = x1.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x1).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, q = kv[0], kv[1]
        pos_encoding = self.positional_encoding
        k = k + pos_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.permute(0, 2, 1), size=x.shape[1], mode='linear').permute(0, 2, 1)
        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, 2, batch_first=True)

    def forward(self, x):
        r_out, (h, c) = self.rnn(x, None)

        return r_out[:, -1, :]

class PoolFormerBlock(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = EntropyWeighting()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers,
                 pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolFormerBlock(
            dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
        ))
    blocks = nn.Sequential(*blocks)

    return blocks

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=(307, 241), patch_size=7, stride=4, in_chans=155, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W

class AIWSEN(nn.Module):
    def __init__(self, layers=[1,1,1,1],img_size=(7,7),in_chans=155, embed_dims=[96,192,384,768],
                 mlp_ratios=[4,4,4,4], downsamples=[True,True,True,True], # , True, True, True
                 pool_size=3,
                 norm_layer=GroupNorm, act_layer=nn.GELU,
                 num_classes=2,
                 in_patch_size=7, in_stride=4, in_pad=2,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-6,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,qkv_bias=False, qk_scale=None,num_stages=2,
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.lstm1 = RNN(in_chans, 32)
        self.lstm2 = RNN(768, 256)
        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=in_chans, embed_dim=embed_dims[0])

        self.FLA = FocusedLinearAttention(
                dim=in_chans, num_patches=49,
                num_heads=5, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=0., proj_drop=0., sr_ratio=1,
                focusing_factor=3, kernel_size=5)

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)
        self.softmax = nn.Softmax(dim=-1)

        if self.fork_feat:
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.fc = nn.Sequential(
            nn.Linear(32+256, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2, bias=True),
        )

        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            outs.append(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward_features(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x
        x = self.norm(x)
        return x

    def forward(self, x1, x2):
        B,C,M,N = x1.shape
        D = x1 - x2
        x11 = x1.permute(0, 2, 3, 1).reshape(B, M * N, C)
        x22 = x2.permute(0, 2, 3, 1).reshape(B, M * N, C)
        X1 = self.FLA(x11, x11, M, N)
        X2 = self.FLA(x22, x22, M, N)
        Cross1 = self.FLA(X1,X2,M,N)
        Cross11 = self.FLA(X2,X1,M,N)
        X11 = self.FLA(X1,X1,M,N)
        X22 = self.FLA(X2,X2,M,N)
        Cross2 = self.FLA(X11, X22, M, N)
        Cross22 = self.FLA(X22, X11, M, N)
        cross = torch.cat([X11 - X22,Cross11 - Cross1,Cross22 - Cross2],dim=1)

        out1 = self.lstm1(cross)
        DDD = self.forward_features(D)
        feature = DDD
        feature = feature.reshape((feature.size(0), feature.size(2) * feature.size(3), feature.size(1)))
        out2 = self.lstm2(feature)
        feature = torch.cat([out1,out2],dim=1)

        deep_out = self.fc(feature)

        final_out = self.softmax(deep_out)

        return final_out

import torch

# 如果你要统计 FLOPs / Params，安装 thop: pip install thop
try:
    from thop import profile
except ImportError:
    profile = None


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ✅ 你的 FLA 写死 num_patches=49，因此必须 H*W=49 => 7x7
    B = 2
    C = 198
    H, W = 7, 7
    num_classes = 2

    model = AIWSEN(
        layers=[1, 1, 1, 1],
        img_size=(H, W),
        in_chans=C,
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 768],
        mlp_ratios=[4, 4, 4, 4],
        downsamples=[True, True, True, True],
        num_stages=2,
    ).to(device)

    model.eval()

    x1 = torch.randn(B, C, H, W, device=device)
    x2 = torch.randn(B, C, H, W, device=device)

    print(f"输入形状: x1={x1.shape}, x2={x2.shape}")

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 前向推理
    with torch.no_grad():
        out = model(x1, x2)

    print(f"输出形状: out={out.shape}")
    print(f"输出示例: {out[:2]}")

    # FLOPs / MACs（可选）
    if profile is not None:
        macs, params = profile(model, inputs=(x1, x2), verbose=False)
        print(f"计算量 MACs：{macs/1e6:.4f} M")
        print(f"计算量 FLOPs：{2*macs/1e6:.4f} M")  # 1 MAC ≈ 2 FLOPs
        print(f"参数量 Params：{params/1e3:.4f} K")
    else:
        print("未安装 thop，跳过 MACs/FLOPs 统计。可执行: pip install thop")
