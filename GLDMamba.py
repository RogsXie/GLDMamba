import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


from VMamba.classification.models.vmamba import VSSBlock


def _bz(*biases):
    """
    将多个 bias 相加，None 会被忽略。
    """
    b = None
    for bi in biases:
        if bi is None:
            continue
        b = bi if b is None else b + bi
    return b


class LDSM(nn.Module):
    """
    Local Differential State Module
    局部差分状态建模模块
    """

    def __init__(self, d):
        super().__init__()

        self.p2 = nn.Conv2d(
            d, d,
            kernel_size=(1, 3),
            padding=(0, 1),
            bias=True
        )

        self.p3 = nn.Conv2d(
            d, d,
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=True
        )

        self.p5 = nn.Conv2d(
            d, d,
            kernel_size=3,
            padding=1,
            bias=True
        )

    def _g2(self):
        """
        HD: Horizontal Difference
        """
        w = self.p2.weight
        o, i, _, _ = w.shape
        v = w.view(o, i, 3)

        w3 = w.new_zeros(o, i, 3, 3)
        w3[:, :, :, 0] = v
        w3[:, :, :, 2] = -v

        return w3, self.p2.bias

    def _g3(self):
        """
        VD: Vertical Difference
        """
        w = self.p3.weight
        o, i, _, _ = w.shape
        v = w.view(o, i, 3)

        w3 = w.new_zeros(o, i, 3, 3)
        w3[:, :, 0, :] = v
        w3[:, :, 2, :] = -v

        return w3, self.p3.bias

    def _g5(self):
        """
        SC: Standard 3x3 Convolution
        """
        return self.p5.weight, self.p5.bias

    def forward(self, x):
        w2, b2 = self._g2()
        w3, b3 = self._g3()
        w5, b5 = self._g5()

        w = torch.stack([w2, w3, w5], dim=0).sum(dim=0)
        b = _bz(b2, b3, b5)

        return F.conv2d(x, w, b, stride=1, padding=1, groups=1)


class LightReduction(nn.Module):
    """
    轻量通道降维模块
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )

    def forward(self, x):
        return self.reduce(x)


class GDSM(nn.Module):
    """
    Global Differential State Module
    全局差分状态建模模块
    """

    def __init__(self, dim=64, drop_path=0.1, d_state=16, mlp_ratio=4.0):
        super().__init__()

        self.block = VSSBlock(
            dim=dim,
            drop_path=drop_path,
            d_state=d_state,
            mlp_ratio=mlp_ratio
        )

    def forward(self, x):
        return self.block(x)


class FFM(nn.Module):
    """
    Feature Fusion Module
    全局-局部差分特征融合模块
    """

    def __init__(self, channels=64, r=4):
        super().__init__()

        inter = channels // r

        self.diff_conv = nn.Sequential(
            nn.Conv2d(channels, inter, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, 3, 1, 1, bias=False),
        )

        self.global_att_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, 1, bias=False),
        )

        self.global_att_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, inter, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        xd = x2 - x1

        xd_feat = self.diff_conv(xd)
        xg_avg = self.global_att_avg(xd)
        xg_max = self.global_att_max(xd)

        xlg = xd_feat + xg_avg + xg_max
        w = self.sigmoid(xlg)

        xo = x1 * w + x2 * (1 - w)

        return xo


class GLDMamba(nn.Module):
    """
    GLDMamba

    输入:
        x1, x2: [H, W, C]

    输出:
        out: [H * W, class_count]
    """

    def __init__(self, height: int, width: int, channel: int, class_count: int):
        super().__init__()

        self.class_count = class_count
        self.channel = channel
        self.height = height
        self.width = width

        # Diff / Cat 浅层特征降维
        self.diff_reduction = LightReduction(channel, 64)
        self.cat_reduction = LightReduction(channel * 2, 64)

        # GDSM: 全局处理路
        self.gdsm_diff = GDSM(
            dim=64,
            drop_path=0.1,
            d_state=16,
            mlp_ratio=4.0
        )

        self.gdsm_cat = GDSM(
            dim=64,
            drop_path=0.1,
            d_state=16,
            mlp_ratio=4.0
        )

        # LDSM: 局部处理路
        self.ldsm_diff = LDSM(64)
        self.ldsm_cat = LDSM(64)

        # 融合模块
        self.fuse_diff = FFM(64)
        self.fuse_cat = FFM(64)

        # 分类头
        final_ch = 64 * 2

        self.head_norm = nn.LayerNorm(final_ch)
        self.head_drop = nn.Dropout(p=0.1)

        self.fc_out = nn.Linear(final_ch, 64)
        self.fc_out1 = nn.Linear(64, 32)
        self.fc_out2 = nn.Linear(32, class_count)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        x1, x2: [H, W, C]
        return: [H * W, class_count]
        """

        # [H, W, C] -> [1, C, H, W]
        x1_input = torch.unsqueeze(x1.permute(2, 0, 1), 0)
        x2_input = torch.unsqueeze(x2.permute(2, 0, 1), 0)

        # Diff / Cat 输入
        xd = torch.abs(x2_input - x1_input)
        xc = torch.abs(torch.cat([x1_input, x2_input], dim=1))

        # 浅层降维
        diff_feat = self.diff_reduction(xd)
        cat_feat = self.cat_reduction(xc)

        # GDSM 全局处理路
        gdsm_diff = self.gdsm_diff(diff_feat)
        gdsm_cat = self.gdsm_cat(cat_feat)

        # LDSM 局部处理路
        ldsm_diff = self.ldsm_diff(diff_feat)
        ldsm_cat = self.ldsm_cat(cat_feat)

        # 交叉融合
        fused_diff = self.fuse_diff(gdsm_diff, ldsm_cat)
        fused_cat = self.fuse_cat(ldsm_diff, gdsm_cat)

        fused_all = torch.cat([fused_diff, fused_cat], dim=1)

        # 逐像素分类: [1, 128, H, W] -> [H * W, 128]
        out = torch.squeeze(fused_all, 0)
        out = out.permute(1, 2, 0)
        out = out.reshape(self.height * self.width, -1)

        out = self.head_norm(out)
        out = self.head_drop(out)

        out = self.fc_out(out)
        out = self.fc_out1(out)
        out = self.fc_out2(out)

        out = F.softmax(out, dim=-1)

        return out


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    height, width, channels = 463, 241, 198
    num_classes = 2

    model = GLDMamba(
        height=height,
        width=width,
        channel=channels,
        class_count=num_classes
    ).to(device)

    x1 = torch.randn(height, width, channels).to(device)
    x2 = torch.randn(height, width, channels).to(device)

    print(f"输入形状: x1={x1.shape}, x2={x2.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    macs, params = profile(
        model,
        inputs=(x1, x2),
        verbose=False
    )

    print(f"计算量 MACs：{macs / 1e6:.4f} M")
    print(f"计算量 FLOPs：{2 * macs / 1e6:.4f} M")
    print(f"参数量 Params：{params / 1e3:.4f} K")

    print("\n开始前向传播测试...")

    model.eval()
    with torch.no_grad():
        out = model(x1, x2)

    print(f"最终输出形状: {out.shape}")
    print(f"输出应该是: ({height * width}, {num_classes})")

    print("\n开始性能测试...")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    iters = 10

    with torch.no_grad():
        for _ in range(iters):
            _ = model(x1, x2)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

    avg = (time.time() - start_time) / iters

    print(f"平均前向传播时间: {avg * 1000:.2f} ms")
    print(f"FPS: {1.0 / avg:.2f}")

    print("\n✅ GLDMamba 测试完成！")
