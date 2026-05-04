import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import sys
import os
from einops.layers.torch import Rearrange
from thop import profile
# 添加 VMamba 目录到 Python 路径
sys.path.append(r'C:\Users\12879\PycharmProjects\vmamba')

# 然后导入 vmamba 模块
from VMamba.classification.models.gldvm import VSSBlock
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def _bias_sum(*biases):
    b = None
    for bi in biases:
        if bi is None:
            continue
        b = bi if b is None else b + bi
    return b

def _bz(*biases):
    """
    将多个 bias 相加，None 会被忽略。
    支持任意个数的 bias，方便后续再加分支。
    """
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

class LightReduction(nn.Module):
    """
    仅保留最基本的降维和对齐功能，不进行深度局部特征提取
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.reduce = nn.Sequential(
            # 1x1 卷积负责改变通道数（降维）
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
            # 这里不再加 3x3 卷积，不让卷积提前“看”太多局部细节
        )

    def forward(self, x):
        return self.reduce(x)

class DepthwiseSeparableConv(nn.Module):
    """DW + PW, 更高效"""
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

class SharedConvBlock(nn.Module):
    """权重共享的卷积块（升级为 DW+PW）"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SharedConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      padding=padding, stride=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ConcatModule(nn.Module):
    """拼接模块: 使用深度可分离卷积（修复并启用）"""
    def __init__(self, in_channels, out_channels):
        super(ConcatModule, self).__init__()
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
        xd = x2 - x1
        xd_feat = self.diff_conv(xd)
        xg_avg = self.global_att_avg(xd)
        xg_max = self.global_att_max(xd)
        xlg = xd_feat + xg_avg + xg_max
        w = self.sigmoid(xlg)
        xo = x1 * w + x2 * (1-w)
        
        return xo

class Net(nn.Module):
    """
    Patch-based 两分支网络
    输入:  x1, x2 -> [B, C, H, W]  (例如 B=32, C=155, H=W=9)
    输出:  logits -> [B, num_classes] (例如 [32, 2])
    """
    def __init__(self, channel: int, class_count: int, patch_size: int = 9):
        super().__init__()
        self.class_count = class_count
        self.channel = channel
        self.patch_size = patch_size

        self.shared_conv3x3 = LightReduction(self.channel, 64)
        self.concat_3x3 = LightReduction(self.channel * 2, 64)
        
        self.vssm_diff1  = VSSBlock(dim=64, drop_path=0.1, d_state=16, mlp_ratio=4.0)
        self.vssm_diff2  = VSSBlock(dim=64, drop_path=0.1, d_state=16, mlp_ratio=4.0)
        self.deconv1 = Localprocess(64)
        self.deconv2 = Localprocess(64)
        self.fusion  = FFM(64)

        # 分类头：GAP -> LN -> MLP -> logits
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

        xd = abs(x2 - x1)
        xc = abs(torch.cat([x1, x2], dim=1))  

        # ------ Diff 分支 ------
        diff_feat = self.shared_conv3x3(xd)          
        # diff_feat = self.shared2_conv3x3(diff_feat)  

        # ------ Concat 分支 ------
        concat_feat = self.concat_3x3(xc)           

        # ------ 全局/局部特征 ------
        global_feat = self.trans1(diff_feat)
        local_feat = self.deconv1(concat_feat)

        global_feat1 = self.trans2(concat_feat)
        local_feat1 = self.deconv2(diff_feat)
        # ------ 融合 ------
        fused_feat1 = self.fusion(global_feat, local_feat1)  
        fused_feat2 = self.fusion(global_feat1,  local_feat)   
        fused_all = torch.cat([fused_feat1, fused_feat2], dim=1) 

        # ------ 分类（样本级）------
        out = self.pool(fused_all).squeeze(-1).squeeze(-1) # [B, 128]
        # out = torch.squeeze(fused_all, 0).permute(1, 2, 0)  # [HW, 128]
        out = self.head_norm(out)
        out = self.head_drop(out)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        logits = self.fc3(out)                         # [B, class_count]

        return logits

if __name__ == "__main__":
    B, C, H, W = 1, 198, 5, 5
    num_classes = 2
    model = Net(channel=C, class_count=num_classes, patch_size=H).cuda() if torch.cuda.is_available() else Net(channel=C, class_count=num_classes, patch_size=H)

    x1 = torch.randn(B, C, H, W).to(next(model.parameters()).device)
    x2 = torch.randn(B, C, H, W).to(next(model.parameters()).device)

    print(f"输入形状: x1={x1.shape}, x2={x2.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    macs, params = profile(model, inputs=(x1, x2), verbose=False)

    print(f"计算量 MACs：{macs / 1e6:.4f} M")
    print(f"计算量 FLOPs：{2 * macs / 1e6:.4f} M")  # 一个 MAC ≈ 2 FLOPs
    print(f"参数量 Params：{params / 1e3:.4f} K")

    print("\n开始前向传播测试...")
    model.eval()
    with torch.no_grad():
        out = model(x1, x2)
    print(f"最终输出形状: {out.shape}")
    # print(f"输出应该是: ({height * width}, {num_classes})")

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

    print("\n✅ 四路结构测试完成！")

    with torch.no_grad():
        y = net(x1, x2)
    print("Input :", x1.shape, x2.shape)
    print("Output:", y.shape)   # -> torch.Size([32, 2])


