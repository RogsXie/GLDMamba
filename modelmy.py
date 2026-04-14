import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import sys
import os
from thop import profile

#修改你的路径
sys.path.append(r'C:\Users\12879\PycharmProjects\vmamba')
from VMamba.classification.models.vmamba import VSSBlock

def _bias_sum(*biases):
    b = None
    for bi in biases:
        if bi is None:
            continue
        b = bi if b is None else b + bi
    return b


class DepthwiseSeparableConv(nn.Module):
    """DW + PW"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ConvBlock(nn.Module):
    """DW+PW 实现的共享卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      padding=padding, stride=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

def bz(*biases):
    b = None
    for bi in biases:
        if bi is None:
            continue
        b = bi if b is None else b + bi
    return b

class Localprocess(nn.Module):
    def __init__(self, d):
        super().__init__()
        # CDC: Center Difference Convolution
        self.p1 = nn.Conv2d(d, d, 3, padding=1, bias=True)
        # HD: Horizontal Difference
        self.p2 = nn.Conv2d(d, d, kernel_size=(1, 3), padding=(0, 1), bias=True)
        # VD: Vertical Difference
        self.p3 = nn.Conv2d(d, d, kernel_size=(3, 1), padding=(1, 0), bias=True)
        # AD: Anti-diagonal Difference
        self.p4 = nn.Conv2d(d, d, 3, padding=1, bias=True)
        # SC: Standard 3×3 Conv (Vanilla)
        self.p5 = nn.Conv2d(d, d, 3, padding=1, bias=True)
        # MDD: Main-diagonal Difference（新增分支）
        self.p6 = nn.Conv2d(d, d, 3, padding=1, bias=True)

    # ----- 各分支 get_weight -----

    def _g1(self):
        """CDC"""
        w = self.p1.weight          # [o, i, 3, 3]
        o, i, _, _ = w.shape
        wf = w.view(o, i, 9)        # flatten 3x3 -> 9
        idx = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8], device=w.device)
        ns = wf.index_select(2, idx).sum(dim=2)   # 邻域和（除中心外）
        wc = wf.clone()
        wc[..., 4] = wf[..., 4] - ns              # 中心减邻域
        return wc.view(o, i, 3, 3), self.p1.bias

    def _g2(self):
        """HD"""
        w = self.p2.weight
        o, i, _, _ = w.shape
        v = w.view(o, i, 3)                       # [o, i, 3]
        w3 = w.new_zeros(o, i, 3, 3)
        w3[:, :, :, 0] = v                        # 左列 +v
        w3[:, :, :, 2] = -v                       # 右列 -v
        return w3, self.p2.bias

    def _g3(self):
        """VD"""
        w = self.p3.weight
        o, i, _, _ = w.shape
        v = w.view(o, i, 3)
        w3 = w.new_zeros(o, i, 3, 3)
        w3[:, :, 0, :] = v                        # 上行 +v
        w3[:, :, 2, :] = -v                       # 下行 -v
        return w3, self.p3.bias

    def _g4(self):
        """AD: anti-diagonal difference"""
        w = self.p4.weight
        o, i, _, _ = w.shape
        wf = w.view(o, i, 9)
        # 反对角方向的重排索引（和你原来的保持一致）
        idx = torch.tensor([3, 0, 1, 6, 4, 2, 7, 8, 5], device=w.device)
        wa = wf - wf.index_select(2, idx)
        return wa.view(o, i, 3, 3), self.p4.bias

    def _g5(self):
        """SC: standard 3x3 conv"""
        return self.p5.weight, self.p5.bias

    def _g6(self):
        """MDD: main-diagonal difference"""
        w = self.p6.weight
        o, i, _, _ = w.shape
        wf = w.view(o, i, 9)
        # 主对角线转置的索引（3x3 转置：[[0,3,6],[1,4,7],[2,5,8]]）
        idx = torch.tensor([0, 3, 6, 1, 4, 7, 2, 5, 8], device=w.device)
        wm = wf - wf.index_select(2, idx)         # 原核 - 转置核
        return wm.view(o, i, 3, 3), self.p6.bias

    # ----- forward -----

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, C, H, W] （和原来完全一致）
        """
        w1, b1 = self._g1()
        w2, b2 = self._g2()
        w3, b3 = self._g3()
        w4, b4 = self._g4()
        w5, b5 = self._g5()
        w6, b6 = self._g6()  # 新增分支

        # 所有 3x3 核做逐元素求和，得到一个融合卷积核
        w = torch.stack([w1, w2, w3, w4, w5, w6], dim=0).sum(dim=0)
        # 所有 bias 相加
        b = _bz(b1, b2, b3, b4, b5, b6)

        return F.conv2d(x, w, b, stride=1, padding=1, groups=1)

class ConcatModule(nn.Module):
    """拼接后压缩特征"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, k=3, s=1, act=True),
            DepthwiseSeparableConv(out_channels, out_channels, k=3, s=1, act=True),
        )

    def forward(self, x):
        return self.conv(x)

class FFM(nn.Module):
    """特征融合模块（局部/全局差分增强）"""
    def __init__(self, channels=64, r=4):
        super().__init__()
        inter = int(channels // r)

        self.diff_conv = nn.Sequential(
            nn.Conv2d(channels, inter, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, 3, 1, 1, bias=False),
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, 1, bias=False),
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
        self.diff_att = nn.Sequential(
            nn.Conv2d(channels, inter, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # xa = torch.abs(x1 + x2)
        # xd = torch.abs(x2 - x1)
        xd = x2 - x1
        xd_feat = self.diff_conv(xd)
        # diff_w = self.diff_att(xd)
        # xl = self.local_att(xa)
        xg_avg = self.global_att_avg(xd)
        xg_max = self.global_att_max(xd)
        xlg = xd_feat + xg_avg + xg_max
        w = self.sigmoid(xlg)
        xo = x1 * w + x2 * (1-w)

        # diff_w = self.diff_att(xd)
        # xo = xo * (1 + diff_w)

        return xo

class Net(nn.Module):
    def __init__(self, height: int, width: int, channel: int, class_count: int):
        super().__init__()
        self.class_count = class_count
        self.channel = channel
        self.height = height
        self.width = width

        self.shared_conv3x3 = ConvBlock(self.channel, 64, kernel_size=3)

        self.concat_3x3 = ConcatModule(self.channel*2 , 64)

        self.vssm_diff1  = VSSBlock(dim=64, drop_path=0.1, d_state=32, mlp_ratio=4.0)
        self.vssm_diff2  = VSSBlock(dim=64, drop_path=0.1, d_state=32, mlp_ratio=4.0)

        self.deconv_diff = Localprocess(64)
        self.deconv_cat  = Localprocess(64)

        self.fuse_diff = FFM(64)
        self.fuse_cat  = FFM(64)

        final_ch = 64*2  # fused_diff(64) + fused_cat(64)
        self.head_norm = nn.LayerNorm(final_ch)
        self.head_drop = nn.Dropout(p=0.1)
        self.fc_out  = nn.Linear(final_ch, 64)
        self.fc_out1 = nn.Linear(64, 32)
        self.fc_out2 = nn.Linear(32, self.class_count)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        # 维度对齐
        x1_input = torch.unsqueeze(x1.permute(2, 0, 1), 0)
        x2_input = torch.unsqueeze(x2.permute(2, 0, 1), 0)

        # 基本分支输入
        xd = abs(x2_input-x1_input)       # Diff 形状: [1, C, H, W]
        xc = abs(torch.cat([x1_input, x2_input], dim=1))   # Cat  形状: [1, 2C, H, W]

        diff_feat = self.shared_conv3x3(xd)   # -> [1, 64, H, W]
        cat_feat  = self.concat_3x3(xc)                             # -> [1, 64, H, W]

        g_diff = self.vssm_diff1(diff_feat)   # [1, 64, H, W]
        g_diff1 = self.deconv_diff(diff_feat) # [1, 64, H, W]
        g_cat = self.vssm_diff2(cat_feat)
        g_cat1  = self.deconv_cat(cat_feat)   # [1, 64, H, W]

        fused_diff = self.fuse_diff(g_diff1, g_cat)   # [1, 64, H, W]
        fused_cat  = self.fuse_cat(g_diff, g_cat1)      # [1, 64, H, W]

        fused_all = torch.cat([fused_diff,fused_cat], dim=1)  # [1, 128, H, W]

        out = torch.squeeze(fused_all , 0).permute(1, 2, 0).reshape(self.height * self.width, -1)  # [HW, 128]
        out = self.head_norm(out)
        out = self.head_drop(out)
        out = self.fc_out(out)
        out = self.fc_out1(out)
        out = self.fc_out2(out)
        out = F.softmax(out, dim=-1)
        return out


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    height, width, channels = 450, 140, 155
    num_classes = 2

    model = Net(height=height, width=width, channel=channels, class_count=num_classes).to(device)

    x1 = torch.randn(height, width, channels).to(device)
    x2 = torch.randn(height, width, channels).to(device)

    print(f"输入形状: x1={x1.shape}, x2={x2.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    macs, params = profile(model, inputs=(x1, x2), verbose=False)

    print(f"计算量 MACs：{macs/1e6:.4f} M")
    print(f"计算量 FLOPs：{2*macs/1e6:.4f} M")   # 一个 MAC ≈ 2 FLOPs
    print(f"参数量 Params：{params/1e3:.4f} K")


    print("\n开始前向传播测试...")
    model.eval()
    with torch.no_grad():
        out = model(x1, x2)
    print(f"最终输出形状: {out.shape}")
    print(f"输出应该是: ({height * width}, {num_classes})")

    import time
    print("\n开始性能测试...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
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


