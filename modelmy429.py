# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import init
# import math
# import sys
# import os
#
# from thop import profile
#
#
#
# # 添加 VMamba 目录到 Python 路径（按你的本地环境，可保留/修改）
# sys.path.append(r'/')
#
# # 然后导入 vmamba 模块
# # from VMamba.classification.models.gldvm import VSSBlock
# from VMamba.classification.models.save import VSSBlock
# # from VMamba.classification.models.vm import VSSBlock as VSSBlock1
# def _bias_sum(*biases):
#     b = None
#     for bi in biases:
#         if bi is None:
#             continue
#         b = bi if b is None else b + bi
#     return b
#
#
# class DepthwiseSeparableConv(nn.Module):
#     """DW + PW"""
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
#
# class ConvBlock(nn.Module):
#     """DW+PW 实现的共享卷积块"""
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super().__init__()
#         padding = kernel_size // 2
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
#                       padding=padding, stride=1, groups=in_channels, bias=False),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return self.conv(x)
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
#         # w = torch.stack([w1, w2, w3, w4, w5, w6], dim=0).sum(dim=0)
#         # 所有 bias 相加
#         # b = _bz(b1, b2, b3, b4, b5, b6)
#
#         w = torch.stack([ w2, w3, w5], dim=0).sum(dim=0)
#         # 所有 bias 相加
#         b = _bz( b2, b3,b5)
#         # w = torch.stack([w5], dim=0).sum(dim=0)
#         # # 所有 bias 相加
#         # b = _bz(b5)
#
#
#
#         return F.conv2d(x, w, b, stride=1, padding=1, groups=1)
# # def  _bz(u1, u2, u3, u4, u5):
# #     return (u1 if u1 is not None else 0) + \
# #            (u2 if u2 is not None else 0) + \
# #            (u3 if u3 is not None else 0) + \
# #            (u4 if u4 is not None else 0) + \
# #            (u5 if u5 is not None else 0)
# #
# #
# # class Localprocess(nn.Module):
# #     def __init__(self, d):
# #         super().__init__()
# #         # CDC
# #         self.p1 = nn.Conv2d(d, d, 3, padding=1, bias=True)
# #         # HD
# #         self.p2 = nn.Conv2d(d, d, kernel_size=(1, 3), padding=(0, 1), bias=True)
# #         # VD
# #         self.p3 = nn.Conv2d(d, d, kernel_size=(3, 1), padding=(1, 0), bias=True)
# #         # AD
# #         self.p4 = nn.Conv2d(d, d, 3, padding=1, bias=True)
# #         # Vanilla 3×3
# #         self.p5 = nn.Conv2d(d, d, 3, padding=1, bias=True)
# #
# #     # ----- 各分支 get_weight -----
# #
# #     def _g1(self):
# #         """CDC"""
# #         w = self.p1.weight
# #         o, i, _, _ = w.shape
# #         wf = w.view(o, i, 9)
# #         idx = torch.tensor([0,1,2,3,5,6,7,8], device=w.device)
# #         ns = wf.index_select(2, idx).sum(dim=2)
# #         wc = wf.clone()
# #         wc[..., 4] = wf[..., 4] - ns
# #         return wc.view(o, i, 3, 3), self.p1.bias
# #
# #     def _g2(self):
# #         """HD"""
# #         w = self.p2.weight
# #         o, i, _, _ = w.shape
# #         v = w.view(o, i, 3)
# #         w3 = w.new_zeros(o, i, 3, 3)
# #         w3[:, :, :, 0] = v
# #         w3[:, :, :, 2] = -v
# #         return w3, self.p2.bias
# #
# #     def _g3(self):
# #         """VD"""
# #         w = self.p3.weight
# #         o, i, _, _ = w.shape
# #         v = w.view(o, i, 3)
# #         w3 = w.new_zeros(o, i, 3, 3)
# #         w3[:, :, 0, :] = v
# #         w3[:, :, 2, :] = -v
# #         return w3, self.p3.bias
# #
# #     def _g4(self):
# #         """AD"""
# #         w = self.p4.weight
# #         o, i, _, _ = w.shape
# #         wf = w.view(o, i, 9)
# #         idx = torch.tensor([3,0,1,6,4,2,7,8,5], device=w.device)
# #         wa = wf - 1.0 * wf.index_select(2, idx)
# #         return wa.view(o, i, 3, 3), self.p4.bias
# #
# #     def _g5(self):
# #         return self.p5.weight, self.p5.bias
# #
# #     # ----- forward -----
# #
# #     def forward(self, x):
# #         w1, b1 = self._g1()
# #         w2, b2 = self._g2()
# #         w3, b3 = self._g3()
# #         w4, b4 = self._g4()
# #         w5, b5 = self._g5()
# #
# #         w = torch.stack([w1, w2, w3, w4, w5], dim=0).sum(dim=0)
# #         b = _bz(b1, b2, b3, b4, b5)
# #
# #         return F.conv2d(x, w, b, stride=1, padding=1, groups=1)
#
#
#
#
#
# class ConcatModule(nn.Module):
#     """拼接后压缩特征"""
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             DepthwiseSeparableConv(in_channels, out_channels, k=3, s=1, act=True),
#             DepthwiseSeparableConv(out_channels, out_channels, k=3, s=1, act=True),
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
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
# class PlainLocalProcess(nn.Module):
#     """
#     将复杂的 Localprocess 替换为标准卷积块。
#     保持与原 Localprocess 相同的输入输出通道数。
#     """
#     def __init__(self, d):
#         super().__init__()
#         # 使用标准的 3x3 卷积，padding=1 保证尺寸不变
#         self.conv = nn.Sequential(
#             nn.Conv2d(d, d, kernel_size=3, padding=1, bias=True),
#             nn.BatchNorm2d(d),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.conv(x)
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
# class Net(nn.Module):
#     def __init__(self, height: int, width: int, channel: int, class_count: int):
#         super().__init__()
#         self.class_count = class_count
#         self.channel = channel
#         self.height = height
#         self.width = width
#         self.shared_conv3x3 = LightReduction(self.channel, 64)
#         self.concat_3x3 = LightReduction(self.channel * 2, 64)
#         # ---- 预处理：Diff 与 Cat 的浅层特征（输出统一到 64 通道）----
#         # self.shared_conv3x3 = ConvBlock(self.channel, 64, kernel_size=3)
#         # self.shared_conv3x32 = ConvBlock(self.channel, 64, kernel_size=3)
#
#         # self.shared2_conv3x3 = ConvBlock(48, 96, kernel_size=3)   # Diff -> 64
#         #
#         # self.tshared_conv3x3 = ConvBlock(self.channel, 48, kernel_size=3)
#         # self.tshared2_conv3x3 = ConvBlock(48, 96, kernel_size=3)   # Diff -> 64
#
#         # self.concat_3x3 = ConcatModule(self.channel*2 , 64)            # Cat  -> 64
#
#         # ---- 全局分支（Mamba）----
#         # self.vssm_diff1 = VSSM(in_chans=64, depths=[1], dims=[64], drop_path=0.1, d_state=32, mlp_ratio=4.0)
#         # self.vssm_diff2 = VSSM(in_chans=64, depths=[1], dims=[64], drop_path=0.1, d_state=32, mlp_ratio=4.0)
#
#         self.vssm_diff1  = VSSBlock(dim=64, drop_path=0.1, d_state=16, mlp_ratio=4.0)
#         self.vssm_diff2  = VSSBlock(dim=64, drop_path=0.1, d_state=16, mlp_ratio=4.0)
#         # self.vssm_diff3 = VSSBlock(dim=64, drop_path=0.1, d_state=32, mlp_ratio=4.0)
#         # self.vssm_diff4 = VSSBlock(dim=64, drop_path=0.1, d_state=32, mlp_ratio=4.0)
#
#         # self.vssm_diff3 = VSSM(in_chans=64, depths=[1], dims=[64], drop_path=0.1, d_state=32, mlp_ratio=4.0)
#         # self.vssm_diff4 = VSSM(in_chans=64, depths=[1], dims=[64], drop_path=0.1, d_state=32, mlp_ratio=4.0)
#
#         # ---- 局部分支（DEConv）----
#         self.deconv_diff = Localprocess(64)
#         self.deconv_cat  = Localprocess(64)
#         # self.deconv_diff = PlainLocalProcess(64)
#         # self.deconv_cat = PlainLocalProcess(64)
#         # self.deconv_diff1 = Localprocess(64)
#         # self.deconv_cat1  = Localprocess(64)
#
#         # ---- 两路分别融合（全局 vs 局部）----
#
#
#         self.fuse_diff = FFM(64)
#         self.fuse_cat  = FFM(64)
#
#         # self.fuse = FFM(64)
#
#         # ---- 分类头 ----
#         final_ch = 64*2 # fused_diff(64) + fused_cat(64)
#         self.head_norm = nn.LayerNorm(final_ch)
#         self.head_drop = nn.Dropout(p=0.1)
#         self.fc_out  = nn.Linear(final_ch, 64)
#         self.fc_out1 = nn.Linear(64, 32)
#         # self.fc_out2 = nn.Linear(32, 16)
#         # self.fc_out3 = nn.Linear(16, 8)
#         self.fc_out2= nn.Linear(32, self.class_count)
#
#         self._initialize_weights()
#
#     def _initialize_weights(self):
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
#         """
#         x1, x2: [H, W, C]
#         输出：[(H*W), num_classes]
#         """
#         # (H, W, C) -> (1, C, H, W)
#
#         # 维度对齐
#         x1_input = torch.unsqueeze(x1.permute(2, 0, 1), 0)
#         x2_input = torch.unsqueeze(x2.permute(2, 0, 1), 0)
#
#
#         # 基本分支输入
#         xd = abs(x2_input-x1_input)       # Diff 形状: [1, C, H, W]
#         xc = abs(torch.cat([x1_input, x2_input], dim=1))   # Cat  形状: [1, 2C, H, W]
#
#         # # 1. 获取通道数
#         # batch, channels, h, w = x1_input.shape
#         # device = x1_input.device
#         #
#         # # 2. 生成随机索引序列 (2 * C)
#         # # 我们将 T1 和 T2 的通道混合在一起进行全随机排列
#         # combined_indices = torch.randperm(2 * channels, device=device)
#         #
#         # # 3. 原始拼接 (1, 2C, H, W)
#         # xc_raw = torch.cat([x1_input, x2_input], dim=1)
#         #
#         # # 4. 根据随机索引重新排列通道 (随机通道拼接)
#         # xc = xc_raw[:, combined_indices, :, :]
#         #
#         # # 5. 保持你原来的绝对值风格
#         # xc = torch.abs(xc)
#
#         # 浅层特征
#
#
#         diff_feat = self.shared_conv3x3(xd)   # -> [1, 64, H, W]
#         cat_feat  = self.concat_3x3(xc)                             # -> [1, 64, H, W]
#         # diff_feat = self.shared_conv3x3(xd)   # -> [1, 64, H, W]
#         # cat_feat  = self.shared_conv3x32(xc)
#         # diff_feat = self.shared_conv3x3(xd)
#         # diff_feat = self.shared_conv3x3(xd)
#         # 全局（Mamba）
#         # g_diff = self.vssm_diff1(torch.cat([diff_feat, cat_feat], dim=1))
#
#
#         g_diff = self.vssm_diff1(diff_feat)   # [1, 64, H, W]
#         # g_diff = self.vssm_diff1(xd)
#         # g_diff = self.vssm_diff3(diff_feat)+g_diff
#         g_diff1 = self.deconv_diff(diff_feat) # [1, 64, H, W]
#
#         g_cat = self.vssm_diff2(cat_feat)
#         # g_cat = self.vssm_diff2(xc)
#         # g_cat = self.vssm_diff4(g_cat)+g_cat
#         g_cat1  = self.deconv_cat(cat_feat)   # [1, 64, H, W]
#         # 两路分别融合
#         # g_cat = abs(g_cat[:, 64:, :, :]-g_cat[:, :64, :, :])  # 前64
#         # g_cat1 = abs(g_cat1[:, 64:, :, :]- g_cat1[:, :64, :, :]) # 后64
#         # cat_diff = cat_first - cat_second  # [1, 64, H, W]
#         # g_diff1 = self.deconv_diff(diff_feat)
#         # g_diff = self.vssm_diff1(g_diff1)   # [1, 64, H, W]
#         # # [1, 64, H, W]
#         # g_cat1 = self.deconv_cat(cat_feat)
#         # g_cat = self.vssm_diff2(g_cat1)
#            # [1, 64, H, W]
#
#         # g_diff3 = self.vssm_diff1(g_diff)   # [1, 64, H, W]
#         # g_diff4 = self.deconv_diff(g_diff1) # [1, 64, H, W]
#         # g_cat3 = self.vssm_diff2(g_cat)
#         # g_cat4  = self.deconv_cat(g_cat1)   # [1, 64, H, W]
#         #
#         # g_diff5 = self.vssm_diff1(g_diff3)   # [1, 64, H, W]
#         # g_diff6 = self.deconv_diff(g_diff4) # [1, 64, H, W]
#         # g_cat5 = self.vssm_diff2(g_cat3)
#         # g_cat6  = self.deconv_cat(g_cat4)   # [1, 64, H, W]
#         #
#         # g_diffa1=g_diff+g_diff3+g_diff5
#         # g_diffa2=g_diff1+g_diff4+g_diff6
#         # g_cata1=g_cat+g_cat3+g_cat5
#         # g_cata2=g_cat1+g_cat4+g_cat6
#
#
#         # fused_diff = self.fuse_diff(g_diffa1, g_cata1)   # [1, 64, H, W]
#         # fused_cat  = self.fuse_cat(g_diffa2, g_cata2)      # [1, 64, H, W]
#
#         # fused_diff = self.fuse_diff(g_diff, g_cat1)   # [1, 64, H, W]
#         # fused_cat  = self.fuse_cat(g_diff1, g_cat)      # [1, 64, H, W]
#         # fused_diff = g_diff+ g_cat1   # [1, 64, H, W]
#         # fused_cat  = g_diff+g_cat     # [1, 64, H, W]
#         # fused_diff = g_diff1 + g_diff1
#         # fused_cat = g_cat + g_cat1
#         fused_diff = self.fuse_diff(g_diff, g_cat1)   # [1, 64, H, W]
#         fused_cat  = self.fuse_cat(g_diff1, g_cat)      # [1, 64, H, W]
#
#
#
#
#         fused_all = torch.cat([fused_diff,fused_cat], dim=1)  # [1, 128, H, W]
#         # fused_all = fused_diff
#         # fused_all = fused_cat
#         # fused_all = torch.cat([g_diff, g_cat], dim=1)  # [1, 128, H, W]
#         # fused_all=g_diff
#
#         # 逐像素分类
#         out = torch.squeeze(fused_all , 0).permute(1, 2, 0).reshape(self.height * self.width, -1)  # [HW, 128]
#         out = self.head_norm(out)
#         out = self.head_drop(out)
#         out = self.fc_out(out)
#         out = self.fc_out1(out)
#         out = self.fc_out2(out)
#         out = F.softmax(out, dim=-1)
#         return out
#
#
# if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     height, width, channels = 463, 241, 198
#     num_classes = 2
#
#     model = Net(height=height, width=width, channel=channels, class_count=num_classes).to(device)
#
#     x1 = torch.randn(height, width, channels).to(device)
#     x2 = torch.randn(height, width, channels).to(device)
#
#     print(f"输入形状: x1={x1.shape}, x2={x2.shape}")
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"总参数量: {total_params:,}")
#     print(f"可训练参数量: {trainable_params:,}")
#
#     macs, params = profile(model, inputs=(x1, x2), verbose=False)
#
#     print(f"计算量 MACs：{macs/1e6:.4f} M")
#     print(f"计算量 FLOPs：{2*macs/1e6:.4f} M")   # 一个 MAC ≈ 2 FLOPs
#     print(f"参数量 Params：{params/1e3:.4f} K")
#
#
#     print("\n开始前向传播测试...")
#     model.eval()
#     with torch.no_grad():
#         out = model(x1, x2)
#     print(f"最终输出形状: {out.shape}")
#     print(f"输出应该是: ({height * width}, {num_classes})")
#
#     import time
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
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

# 添加 VMamba 目录到 Python 路径
sys.path.append(r'/')

from VMambaold.classification.models.save import VSSBlock


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