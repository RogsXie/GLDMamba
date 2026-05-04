import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange


# -----------------------------
# Embedding: 修改为支持动态 window_size
# -----------------------------
class PixelEmbedding(nn.Module):
    def __init__(self, num_bands, embed_dim, window_size=6):  # 默认改为 6
        super(PixelEmbedding, self).__init__()
        self.fc = nn.Linear(num_bands, embed_dim)
        self.window_size = window_size
        # 位置编码长度随 window_size 平方改变
        self.positional_encoding = nn.Parameter(torch.randn(1, window_size * window_size, embed_dim))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

    def forward(self, x):
        # 自动处理 [B, C, H, W] -> [B, L, C]
        if x.ndim == 4:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        x = self.fc(x)
        x += self.positional_encoding
        return x


# -----------------------------
# Utils: DropPath
# -----------------------------
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# -----------------------------
# GLA: Global-Local Attention
# -----------------------------
class GLA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=3,
                 alpha=0.5):
        super().__init__()
        head_dim = dim // num_heads
        self.dim = dim
        self.l_heads = int(num_heads * alpha)
        self.l_dim = self.l_heads * head_dim
        self.h_heads = num_heads - self.l_heads
        self.h_dim = self.h_heads * head_dim
        self.ws = window_size
        self.scale = qk_scale or head_dim ** -0.5

        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    def loc(self, x):
        B, H, W, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1,
                                                                                                              4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, H, W, self.h_dim)
        return self.h_proj(x)

    def glo(self, x):
        B, H, W, C = x.shape
        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)
        if self.ws > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        return self.l_proj(x)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.reshape(B, H, W, C)
        if self.h_heads == 0: return self.glo(x).reshape(B, N, C)
        if self.l_heads == 0: return self.loc(x).reshape(B, N, C)
        x = torch.cat((self.loc(x), self.glo(x)), dim=-1)
        return x.reshape(B, N, C)


# -----------------------------
# FeedForward: 修改 window_size 为变量
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, window_size=6, ffn_expansion_factor=2.66, bias=True):
        super(FeedForward, self).__init__()
        self.window_size = window_size
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        B, L, C = x.shape
        ws = self.window_size
        x = rearrange(x, 'b (h w) c -> b c h w', h=ws, w=ws)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x2) * x1 + F.gelu(x1) * x2
        x = self.project_out(x)
        return rearrange(x, 'b c h w -> b (h w) c')


# -----------------------------
# Block
# -----------------------------
class Block(nn.Module):
    def __init__(self, dim, num_heads, window_size=6, mlp_ratio=2.66, qkv_bias=False, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0., norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        # 注意：这里的 window_size=3 是 GLA 内部提取局部的窗口大小，通常保持 3 即可，不需要和全局 window_size 一致
        # self.gla = GLA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                attn_drop=attn_drop_ratio, proj_drop=drop_ratio, window_size=3, alpha=0.5)

        self.gla = GLA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                       attn_drop=attn_drop_ratio, proj_drop=drop_ratio,
                       window_size=window_size, alpha=0.5)  # 修改这里
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.ffn = FeedForward(dim, window_size=window_size, ffn_expansion_factor=mlp_ratio)

    def forward(self, x):
        ws = self.window_size
        x = x + self.drop_path(self.gla(self.norm1(x), ws, ws))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# -----------------------------
# GLAFormer: 统一修改为 window_size=6
# -----------------------------
class GLAFormer(nn.Module):
    def __init__(self, img_c, num_classes=2, window_size=6,  # 默认设为 6
                 embed_dim=128, depth=4, num_heads=8, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.):
        super(GLAFormer, self).__init__()
        self.window_size = window_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Embedding 层
        self.img1_patch_embed = PixelEmbedding(num_bands=img_c, embed_dim=embed_dim, window_size=window_size)
        self.img2_patch_embed = PixelEmbedding(num_bands=img_c, embed_dim=embed_dim, window_size=window_size)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]

        # Branch 1
        self.img1_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                  attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.img1_norm = norm_layer(embed_dim)

        # Branch 2
        self.img2_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                  attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.img2_norm = norm_layer(embed_dim)

        # Head
        self.conv1 = nn.Sequential(nn.Conv2d(embed_dim, 16, 3, padding=1), nn.ReLU())
        self.conv2 = nn.Conv2d(16, 1, 3, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(window_size * window_size, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes)
        )
        self.apply(_init_vit_weights)

    def forward(self, img1, img2):
        ws = self.window_size

        # Branch 1
        x1 = self.img1_patch_embed(img1)
        x1 = self.img1_blocks(x1)
        x1 = self.img1_norm(x1)

        # Branch 2
        x2 = self.img2_patch_embed(img2)  # 修正了原代码可能存在的变量引用错误
        x2 = self.img2_blocks(x2)
        x2 = self.img2_norm(x2)

        # Fusion
        fuse = torch.abs(x1 - x2)
        B, L, C = fuse.shape

        # [B, L, C] -> [B, C, ws, ws]
        x = fuse.transpose(1, 2).reshape(B, C, ws, ws)
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(B, -1)
        return self.fc(x)


# -----------------------------
# 测试代码 (验证 6x6 是否生效)
# -----------------------------
if __name__ == "__main__":
    B = 1
    ws = 6  # 目标窗口大小
    C_in = 155

    model = GLAFormer(img_c=C_in, window_size=ws, embed_dim=256).cuda()

    # 模拟输入 [B, L, C] 其中 L = 6*6 = 36
    x1 = torch.randn(B, ws * ws, C_in).cuda()
    x2 = torch.randn(B, ws * ws, C_in).cuda()

    out = model(x1, x2)
    print(f"输入形状: {x1.shape}")
    print(f"输出形状: {out.shape}")  # 应为 [1, 2]
