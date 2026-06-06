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

        self.p1 = nn.Conv2d(d, d, 3, padding=1, bias=True)          # CDC
        self.p2 = nn.Conv2d(d, d, kernel_size=(1, 3), padding=(0, 1), bias=True)  # HD
        self.p3 = nn.Conv2d(d, d, kernel_size=(3, 1), padding=(1, 0), bias=True)  # VD
        self.p4 = nn.Conv2d(d, d, 3, padding=1, bias=True)          # AD
        self.p5 = nn.Conv2d(d, d, 3, padding=1, bias=True)          # SC
        self.p6 = nn.Conv2d(d, d, 3, padding=1, bias=True)          # MDD

    def _g1(self):
        """CDC: Center Difference Convolution"""
        w = self.p1.weight
        o, i, _, _ = w.shape
        wf = w.view(o, i, 9)

        idx = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8], device=w.device)
        ns = wf.index_select(2, idx).sum(dim=2)

        wc = wf.clone()
        wc[..., 4] = wf[..., 4] - ns

        return wc.view(o, i, 3, 3), self.p1.bias

    def _g2(self):
        """HD: Horizontal Difference"""
        w = self.p2.weight
        o, i, _, _ = w.shape
        v = w.view(o, i, 3)

        w3 = w.new_zeros(o, i, 3, 3)
        w3[:, :, :, 0] = v
        w3[:, :, :, 2] = -v

        return w3, self.p2.bias

    def _g3(self):
        """VD: Vertical Difference"""
        w = self.p3.weight
        o, i, _, _ = w.shape
        v = w.view(o, i, 3)

        w3 = w.new_zeros(o, i, 3, 3)
        w3[:, :, 0, :] = v
        w3[:, :, 2, :] = -v

        return w3, self.p3.bias

    def _g4(self):
        """AD: Anti-diagonal Difference"""
        w = self.p4.weight
        o, i, _, _ = w.shape
        wf = w.view(o, i, 9)

        idx = torch.tensor([3, 0, 1, 6, 4, 2, 7, 8, 5], device=w.device)
        wa = wf - wf.index_select(2, idx)

        return wa.view(o, i, 3, 3), self.p4.bias

    def _g5(self):
        """SC: Standard Convolution"""
        return self.p5.weight, self.p5.bias

    def _g6(self):
        """MDD: Main-diagonal Difference"""
        w = self.p6.weight
        o, i, _, _ = w.shape
        wf = w.view(o, i, 9)

        idx = torch.tensor([0, 3, 6, 1, 4, 7, 2, 5, 8], device=w.device)
        wm = wf - wf.index_select(2, idx)

        return wm.view(o, i, 3, 3), self.p6.bias

    def forward(self, x):
        w1, b1 = self._g1()
        w2, b2 = self._g2()
        w3, b3 = self._g3()
        w4, b4 = self._g4()
        w5, b5 = self._g5()
        w6, b6 = self._g6()

        w = torch.stack([w1, w2, w3, w4, w5, w6], dim=0).sum(dim=0)
        b = _bz(b1, b2, b3, b4, b5, b6)

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
    全局-局部特征融合模块
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


class GLDMambaP(nn.Module):
    """
    GLDMamba

    输入:
        x1, x2: [B, C, H, W]

    输出:
        logits: [B, class_count]
    """

    def __init__(self, channel: int, class_count: int, patch_size: int = 9):
        super().__init__()

        self.class_count = class_count
        self.channel = channel
        self.patch_size = patch_size

        # 差分分支和拼接分支的轻量降维
        self.diff_reduction = LightReduction(channel, 64)
        self.concat_reduction = LightReduction(channel * 2, 64)

        # GDSM: 全局处理路
        self.gdsm1 = GDSM(dim=64, drop_path=0.1, d_state=16, mlp_ratio=4.0)
        self.gdsm2 = GDSM(dim=64, drop_path=0.1, d_state=16, mlp_ratio=4.0)

        # LDSM: 局部处理路
        self.ldsm1 = LDSM(64)
        self.ldsm2 = LDSM(64)

        # 特征融合
        self.fusion = FFM(64)

        # 分类头
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_norm = nn.LayerNorm(128)
        self.head_drop = nn.Dropout(p=0.1)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, class_count)

        self._init_weights()

    def _init_weights(self):
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
        # x1, x2: [B, C, H, W]

        xd = torch.abs(x2 - x1)
        xc = torch.abs(torch.cat([x1, x2], dim=1))

        # 差分分支
        diff_feat = self.diff_reduction(xd)

        # 拼接分支
        concat_feat = self.concat_reduction(xc)

        # GDSM 全局处理
        global_feat1 = self.gdsm1(diff_feat)
        global_feat2 = self.gdsm2(concat_feat)

        # LDSM 局部处理
        local_feat1 = self.ldsm1(concat_feat)
        local_feat2 = self.ldsm2(diff_feat)

        # 交叉融合
        fused_feat1 = self.fusion(global_feat1, local_feat1)
        fused_feat2 = self.fusion(global_feat2, local_feat2)

        fused_all = torch.cat([fused_feat1, fused_feat2], dim=1)

        # 分类
        out = self.pool(fused_all).squeeze(-1).squeeze(-1)
        out = self.head_norm(out)
        out = self.head_drop(out)

        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        logits = self.fc3(out)

        return logits


if __name__ == "__main__":

    B, C, H, W = 1, 198, 5, 5
    num_classes = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GLDMambaP(
        channel=C,
        class_count=num_classes,
        patch_size=H
    ).to(device)

    x1 = torch.randn(B, C, H, W).to(device)
    x2 = torch.randn(B, C, H, W).to(device)

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

    print(f"计算量 MACs: {macs / 1e6:.4f} M")
    print(f"计算量 FLOPs: {2 * macs / 1e6:.4f} M")
    print(f"参数量 Params: {params / 1e3:.4f} K")

    print("\n开始前向传播测试...")

    model.eval()
    with torch.no_grad():
        out = model(x1, x2)

    print(f"最终输出形状: {out.shape}")

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

    print("\nGLDMamba 测试完成！")
