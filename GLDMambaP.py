# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import init
# import math
# import sys
# import os
# from einops.layers.torch import Rearrange
# from thop import profile
# # 添加 VMamba 目录到 Python 路径
# sys.path.append(r'C:\Users\12879\PycharmProjects\vmamba')
#
# # 然后导入 vmamba 模块
# # from VMamba.classification.models.gldvm import VSSBlock
# from VMamba.classification.models.vm import VSSBlock
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#
#
# def _bias_sum(*biases):
#     b = None
#     for bi in biases:
#         if bi is None:
#             continue
#         b = bi if b is None else b + bi
#     return b
#
# def _bz(*biases):
#     """
#     将多个 bias 相加，None 会被忽略。
#     支持任意个数的 bias，方便后续再加分支。
#     """
#     b = None
#     for bi in biases:
#         if bi is None:
#             continue
#         b = bi if b is None else b + bi
#     return b
#
# class Localprocess(nn.Module):
#     def __init__(self, d):
#         super().__init__()
#         # CDC: Center Difference Convolution
#         self.p1 = nn.Conv2d(d, d, 3, padding=1, bias=True)
#         # HD: Horizontal Difference
#         self.p2 = nn.Conv2d(d, d, kernel_size=(1, 3), padding=(0, 1), bias=True)
#         # VD: Vertical Difference
#         self.p3 = nn.Conv2d(d, d, kernel_size=(3, 1), padding=(1, 0), bias=True)
#         # AD: Anti-diagonal Difference
#         self.p4 = nn.Conv2d(d, d, 3, padding=1, bias=True)
#         # SC: Standard 3×3 Conv (Vanilla)
#         self.p5 = nn.Conv2d(d, d, 3, padding=1, bias=True)
#         # MDD: Main-diagonal Difference（新增分支）
#         self.p6 = nn.Conv2d(d, d, 3, padding=1, bias=True)
#
#     # ----- 各分支 get_weight -----
#
#     def _g1(self):
#         """CDC"""
#         w = self.p1.weight          # [o, i, 3, 3]
#         o, i, _, _ = w.shape
#         wf = w.view(o, i, 9)        # flatten 3x3 -> 9
#         idx = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8], device=w.device)
#         ns = wf.index_select(2, idx).sum(dim=2)   # 邻域和（除中心外）
#         wc = wf.clone()
#         wc[..., 4] = wf[..., 4] - ns              # 中心减邻域
#         return wc.view(o, i, 3, 3), self.p1.bias
#
#     def _g2(self):
#         """HD"""
#         w = self.p2.weight
#         o, i, _, _ = w.shape
#         v = w.view(o, i, 3)                       # [o, i, 3]
#         w3 = w.new_zeros(o, i, 3, 3)
#         w3[:, :, :, 0] = v                        # 左列 +v
#         w3[:, :, :, 2] = -v                       # 右列 -v
#         return w3, self.p2.bias
#
#     def _g3(self):
#         """VD"""
#         w = self.p3.weight
#         o, i, _, _ = w.shape
#         v = w.view(o, i, 3)
#         w3 = w.new_zeros(o, i, 3, 3)
#         w3[:, :, 0, :] = v                        # 上行 +v
#         w3[:, :, 2, :] = -v                       # 下行 -v
#         return w3, self.p3.bias
#
#     def _g4(self):
#         """AD: anti-diagonal difference"""
#         w = self.p4.weight
#         o, i, _, _ = w.shape
#         wf = w.view(o, i, 9)
#         # 反对角方向的重排索引（和你原来的保持一致）
#         idx = torch.tensor([3, 0, 1, 6, 4, 2, 7, 8, 5], device=w.device)
#         wa = wf - wf.index_select(2, idx)
#         return wa.view(o, i, 3, 3), self.p4.bias
#
#     def _g5(self):
#         """SC: standard 3x3 conv"""
#         return self.p5.weight, self.p5.bias
#
#     def _g6(self):
#         """MDD: main-diagonal difference"""
#         w = self.p6.weight
#         o, i, _, _ = w.shape
#         wf = w.view(o, i, 9)
#         # 主对角线转置的索引（3x3 转置：[[0,3,6],[1,4,7],[2,5,8]]）
#         idx = torch.tensor([0, 3, 6, 1, 4, 7, 2, 5, 8], device=w.device)
#         wm = wf - wf.index_select(2, idx)         # 原核 - 转置核
#         return wm.view(o, i, 3, 3), self.p6.bias
#
#     # ----- forward -----
#
#     def forward(self, x):
#         """
#         x: [B, C, H, W]
#         return: [B, C, H, W] （和原来完全一致）
#         """
#         w1, b1 = self._g1()
#         w2, b2 = self._g2()
#         w3, b3 = self._g3()
#         w4, b4 = self._g4()
#         w5, b5 = self._g5()
#         w6, b6 = self._g6()  # 新增分支
#
#         # 所有 3x3 核做逐元素求和，得到一个融合卷积核
#         w = torch.stack([w1, w2, w3, w4, w5, w6], dim=0).sum(dim=0)
#         # 所有 bias 相加
#         b = _bz(b1, b2, b3, b4, b5, b6)
#
#         return F.conv2d(x, w, b, stride=1, padding=1, groups=1)
#
# class LightReduction(nn.Module):
#     """
#     仅保留最基本的降维和对齐功能，不进行深度局部特征提取
#     """
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.reduce = nn.Sequential(
#             # 1x1 卷积负责改变通道数（降维）
#             nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.1)
#             # 这里不再加 3x3 卷积，不让卷积提前“看”太多局部细节
#         )
#
#     def forward(self, x):
#         return self.reduce(x)
#
# class DepthwiseSeparableConv(nn.Module):
#     """DW + PW, 更高效"""
#     def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=True):
#         super().__init__()
#         if p is None:
#             p = k // 2
#         self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
#         self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
#         self.bn = nn.BatchNorm2d(out_ch)
#         self.act = nn.ReLU(inplace=True) if act else nn.Identity()
#
#     def forward(self, x):
#         x = self.dw(x)
#         x = self.pw(x)
#         x = self.bn(x)
#         x = self.act(x)
#         return x
#
# class SharedConvBlock(nn.Module):
#     """权重共享的卷积块（升级为 DW+PW）"""
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super(SharedConvBlock, self).__init__()
#         padding = kernel_size // 2
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
#                       padding=padding, stride=1, groups=in_channels, bias=False),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
# class ConcatModule(nn.Module):
#     """拼接模块: 使用深度可分离卷积（修复并启用）"""
#     def __init__(self, in_channels, out_channels):
#         super(ConcatModule, self).__init__()
#         self.conv = nn.Sequential(
#             DepthwiseSeparableConv(in_channels, out_channels, k=3, s=1, act=True),
#             DepthwiseSeparableConv(out_channels, out_channels, k=3, s=1, act=True),
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
# class FFM(nn.Module):
#     """特征融合模块（局部/全局差分增强）"""
#     def __init__(self, channels=64, r=4):
#         super().__init__()
#         inter = int(channels // r)
#
#         self.diff_conv = nn.Sequential(
#             nn.Conv2d(channels, inter, 3, 1, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter, channels, 3, 1, 1, bias=False),
#         )
#         self.local_att = nn.Sequential(
#             nn.Conv2d(channels, inter, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter, channels, 1, bias=False),
#         )
#         self.global_att_avg = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, inter, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter, channels, 1, bias=False),
#         )
#         self.global_att_max = nn.Sequential(
#             nn.AdaptiveMaxPool2d(1),
#             nn.Conv2d(channels, inter, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter, channels, 1, bias=False),
#         )
#         self.diff_att = nn.Sequential(
#             nn.Conv2d(channels, inter, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(inter, channels, 1, bias=False),
#             nn.Sigmoid(),
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x1, x2):
#         # xa = torch.abs(x1 + x2)
#         # xd = torch.abs(x2 - x1)
#         xd = x2 - x1
#         xd_feat = self.diff_conv(xd)
#         # diff_w = self.diff_att(xd)
#         # xl = self.local_att(xa)
#         xg_avg = self.global_att_avg(xd)
#         xg_max = self.global_att_max(xd)
#         xlg = xd_feat + xg_avg + xg_max
#         w = self.sigmoid(xlg)
#         xo = x1 * w + x2 * (1-w)
#
#         # diff_w = self.diff_att(xd)
#         # xo = xo * (1 + diff_w)
#
#         return xo
#
#
# class MultiHeadSelfAttention(nn.Module):
#     """
#     2D 空间自注意力：[B, C, H, W] → flatten tokens → MHSA → reshape 回 [B, C, H, W]
#     """
#
#     def __init__(self, dim: int, num_heads: int = 4,
#                  attn_drop: float = 0.0, proj_drop: float = 0.0):
#         super().__init__()
#         assert dim % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=True)
#         self.proj = nn.Linear(dim, dim, bias=True)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, C, H, W = x.shape
#         N = H * W
#         x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]
#
#         qkv = self.qkv(x_flat)  # [B, N, 3C]
#         qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
#         qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, h, N, d]
#         q, k, v = qkv.unbind(0)  # each [B, h, N, d]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, h, N, N]
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
#         out = self.proj_drop(self.proj(out))
#         return out.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
#
#
# class TransformerFFN(nn.Module):
#     """逐点 FFN（1×1 卷积实现，保留空间维度）"""
#
#     def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
#         super().__init__()
#         hidden = int(dim * mlp_ratio)
#         self.net = nn.Sequential(
#             nn.Conv2d(dim, hidden, 1, bias=True),
#             nn.GELU(),
#             nn.Dropout(drop),
#             nn.Conv2d(hidden, dim, 1, bias=True),
#             nn.Dropout(drop),
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# class SpatialTransformerBlock(nn.Module):
#     """
#     Pre-LN Transformer Block:
#         LN → MHSA → residual
#         LN → FFN  → residual
#     LN 施加在 channel 维（[B,C,H,W] permute 后做 LayerNorm）
#     """
#
#     def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0,
#                  drop: float = 0.0, attn_drop: float = 0.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = MultiHeadSelfAttention(dim, num_heads, attn_drop, drop)
#         self.norm2 = nn.LayerNorm(dim)
#         self.ffn = TransformerFFN(dim, mlp_ratio, drop)
#
#     def _ln(self, x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
#         """[B,C,H,W] → LayerNorm on C → [B,C,H,W]"""
#         B, C, H, W = x.shape
#         return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x + self.attn(self._ln(x, self.norm1))
#         x = x + self.ffn(self._ln(x, self.norm2))
#         return x
#
#
# class SpatialTransformer(nn.Module):
#     """
#     堆叠若干 SpatialTransformerBlock。
#     接口与原 VSSBlock 完全兼容：输入/输出均为 [B, dim, H, W]
#     """
#
#     def __init__(self, dim: int, depth: int = 1, num_heads: int = 4,
#                  mlp_ratio: float = 4.0, drop: float = 0.0,
#                  attn_drop: float = 0.0, drop_path: float = 0.0):
#         super().__init__()
#         self.blocks = nn.ModuleList([
#             SpatialTransformerBlock(dim, num_heads, mlp_ratio, drop, attn_drop)
#             for _ in range(depth)
#         ])
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         for blk in self.blocks:
#             x = blk(x)
#         return x
#
# class Net(nn.Module):
#     """
#     Patch-based 两分支网络
#     输入:  x1, x2 -> [B, C, H, W]  (例如 B=32, C=155, H=W=9)
#     输出:  logits -> [B, num_classes] (例如 [32, 2])
#     """
#     def __init__(self, channel: int, class_count: int, patch_size: int = 9):
#         super().__init__()
#         self.class_count = class_count
#         self.channel = channel
#         self.patch_size = patch_size
#
#         # Diff 分支: (x2-x1) -> 32 -> 64
#         # self.shared_conv3x3  = SharedConvBlock(self.channel, 32, kernel_size=3)
#         # self.shared2_conv3x3 = SharedConvBlock(32, 64, kernel_size=3)
#         #
#         # # Concat 分支: cat(x1, x2) -> 64
#         # self.concat_3x3 = ConcatModule(self.channel * 2, 64)
#         self.shared_conv3x3 = LightReduction(self.channel, 64)
#         self.concat_3x3 = LightReduction(self.channel * 2, 64)
#         # 全局/局部模块
#         # self.vssm1  = VSSM(in_chans=64, depths=[1], dims=[64], drop_path=0.1, d_state=32, mlp_ratio=4.0)
#         # self.vssm2  = VSSM(in_chans=64, depths=[1], dims=[64], drop_path=0.1, d_state=32, mlp_ratio=4.0)
#         self.vssm_diff1  = VSSBlock(dim=64, drop_path=0.1, d_state=16, mlp_ratio=4.0)
#         self.vssm_diff2  = VSSBlock(dim=64, drop_path=0.1, d_state=16, mlp_ratio=4.0)
#         self.trans1 = SpatialTransformer(dim=64, depth=1, num_heads=4, mlp_ratio=4.0, drop=0.1)
#         self.trans2 = SpatialTransformer(dim=64, depth=1, num_heads=4, mlp_ratio=4.0, drop=0.1)
#         self.deconv1 = Localprocess(64)
#         self.deconv2 = Localprocess(64)
#         self.fusion  = FFM(64)
#
#         # 分类头：GAP -> LN -> MLP -> logits
#         self.pool = nn.AdaptiveAvgPool2d(1)          # (B, 128, 1, 1)
#         self.head_norm = nn.LayerNorm(128)           # 对通道维做 LN
#         self.head_drop = nn.Dropout(p=0.1)
#         self.fc1 = nn.Linear(128, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, class_count)
#
#         self._init_weights()
#
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 if m.weight is not None:
#                     nn.init.constant_(m.weight, 1)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.trunc_normal_(m.weight, std=0.02)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def forward(self, x1: torch.Tensor, x2: torch.Tensor):
#         # 期望输入: x1, x2 -> [B, C, H, W]
#         # 差分和拼接
#         xd = abs(x2 - x1)
#         xc = abs(torch.cat([x1, x2], dim=1))   # [B, 2C, H, W]
#
#         # ------ Diff 分支 ------
#         diff_feat = self.shared_conv3x3(xd)          # [B, 32, H, W]
#         # diff_feat = self.shared2_conv3x3(diff_feat)  # [B, 64, H, W]
#
#         # ------ Concat 分支 ------
#         concat_feat = self.concat_3x3(xc)            # [B, 64, H, W]
#
#         # ------ 全局/局部特征 ------
#         # global_feat = self.vssm_diff1(diff_feat)          # [B, 64, H, W]
#         # local_feat  = self.deconv1(concat_feat)      # [B, 64, H, W]
#         # global_feat1 = self.vssm_diff2(concat_feat)          # [B, 64, H, W]
#         # local_feat1  = self.deconv2(diff_feat)      # [B, 64, H, W]
#         global_feat = self.trans1(diff_feat)
#         local_feat = self.deconv1(concat_feat)
#         #
#         global_feat1 = self.trans2(concat_feat)
#         local_feat1 = self.deconv2(diff_feat)
#         # ------ 融合 ------
#         fused_feat1 = self.fusion(global_feat, local_feat1)  # [B, 64, H, W]
#         fused_feat2 = self.fusion(global_feat1,  local_feat)    # [B, 64, H, W]
#         fused_all = torch.cat([fused_feat1, fused_feat2], dim=1) # [B, 128, H, W]
#
#         # ------ 分类（样本级）------
#         out = self.pool(fused_all).squeeze(-1).squeeze(-1) # [B, 128]
#         # out = torch.squeeze(fused_all, 0).permute(1, 2, 0)  # [HW, 128]
#         out = self.head_norm(out)
#         out = self.head_drop(out)
#         out = F.relu(self.fc1(out), inplace=True)
#         out = F.relu(self.fc2(out), inplace=True)
#         logits = self.fc3(out)                         # [B, class_count]
#
#         # 训练建议直接返回 logits，用 CrossEntropyLoss
#         return logits
#
# if __name__ == "__main__":
#     B, C, H, W = 1, 198, 5, 5
#     num_classes = 2
#     model = Net(channel=C, class_count=num_classes, patch_size=H).cuda() if torch.cuda.is_available() else Net(channel=C, class_count=num_classes, patch_size=H)
#
#     x1 = torch.randn(B, C, H, W).to(next(model.parameters()).device)
#     x2 = torch.randn(B, C, H, W).to(next(model.parameters()).device)
#
#     print(f"输入形状: x1={x1.shape}, x2={x2.shape}")
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"总参数量: {total_params:,}")
#     print(f"可训练参数量: {trainable_params:,}")
#
#     macs, params = profile(model, inputs=(x1, x2), verbose=False)
#
#     print(f"计算量 MACs：{macs / 1e6:.4f} M")
#     print(f"计算量 FLOPs：{2 * macs / 1e6:.4f} M")  # 一个 MAC ≈ 2 FLOPs
#     print(f"参数量 Params：{params / 1e3:.4f} K")
#
#     print("\n开始前向传播测试...")
#     model.eval()
#     with torch.no_grad():
#         out = model(x1, x2)
#     print(f"最终输出形状: {out.shape}")
#     # print(f"输出应该是: ({height * width}, {num_classes})")
#
#     import time
#
#     print("\n开始性能测试...")
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     start_time = time.time()
#     iters = 10
#     with torch.no_grad():
#         for _ in range(iters):
#             _ = model(x1, x2)
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#     avg = (time.time() - start_time) / iters
#     print(f"平均前向传播时间: {avg * 1000:.2f} ms")
#     print(f"FPS: {1.0 / avg:.2f}")
#
#     print("\n✅ 四路结构测试完成！")
#
#     with torch.no_grad():
#         y = net(x1, x2)
#     print("Input :", x1.shape, x2.shape)
#     print("Output:", y.shape)   # -> torch.Size([32, 2])
#
#
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

# 添加 VMamba 目录到 Python 路径
sys.path.append(r'C:\Users\12879\PycharmProjects\vmamba')

from VMambaold.classification.models.vm import VSSBlock


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
