# # glformer.py
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# #
# #
# # # -----------------------------
# # # Utils
# # # -----------------------------
# # def drop_path(x, drop_prob: float = 0.0, training: bool = False):
# #     if drop_prob == 0.0 or not training:
# #         return x
# #     keep_prob = 1 - drop_prob
# #     shape = (x.shape[0],) + (1,) * (x.ndim - 1)
# #     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
# #     random_tensor.floor_()
# #     return x.div(keep_prob) * random_tensor
# #
# #
# # class DropPath(nn.Module):
# #     def __init__(self, drop_prob=None):
# #         super().__init__()
# #         self.drop_prob = drop_prob
# #
# #     def forward(self, x):
# #         return drop_path(x, self.drop_prob, self.training)
# #
# #
# # def _init_vit_weights(m):
# #     if isinstance(m, nn.Linear):
# #         nn.init.trunc_normal_(m.weight, std=0.01)
# #         if m.bias is not None:
# #             nn.init.zeros_(m.bias)
# #     elif isinstance(m, nn.Conv2d):
# #         nn.init.kaiming_normal_(m.weight, mode="fan_out")
# #         if m.bias is not None:
# #             nn.init.zeros_(m.bias)
# #     elif isinstance(m, nn.LayerNorm):
# #         nn.init.zeros_(m.bias)
# #         nn.init.ones_(m.weight)
# #
# #
# # # -----------------------------
# # # Embedding
# # # -----------------------------
# # class PixelEmbedding(nn.Module):
# #     """
# #     将每个像素的多光谱/多通道向量 (num_bands) 映射到 embed_dim，并加上固定窗口内的位置编码。
# #     输入:  [B, L=window_size*window_size, num_bands]
# #     输出:  [B, L, embed_dim]
# #     """
# #     def __init__(self, num_bands: int, embed_dim: int, window_size: int):
# #         super().__init__()
# #         self.window_size = window_size
# #         self.fc = nn.Linear(num_bands, embed_dim)
# #         self.positional_encoding = nn.Parameter(
# #             torch.zeros(1, window_size * window_size, embed_dim)
# #         )
# #         nn.init.trunc_normal_(self.positional_encoding, std=0.02)
# #
# #     def forward(self, x):
# #         x = self.fc(x)  # [B, L, D]
# #         x = x + self.positional_encoding  # broadcast
# #         return x
# #
# #
# # # -----------------------------
# # # GLA: Global-Local Attention
# # # -----------------------------
# # class GLA(nn.Module):
# #     """
# #     将注意力头划分为局部与全局两部分，局部在 (ws x ws) 窗格内做注意力，全局做降采样的注意力。
# #     期望输入: [B, N, C]，其中 N = H*W
# #     forward(x, H, W)
# #     """
# #     def __init__(
# #         self,
# #         dim: int,
# #         num_heads: int = 8,
# #         qkv_bias: bool = False,
# #         qk_scale=None,
# #         attn_drop: float = 0.0,
# #         proj_drop: float = 0.0,
# #         window_size: int = 3,
# #         alpha: float = 0.5,
# #     ):
# #         super().__init__()
# #         assert dim % num_heads == 0, "dim must be divisible by num_heads"
# #         head_dim = dim // num_heads
# #
# #         self.dim = dim
# #         self.ws = window_size
# #         self.scale = qk_scale or head_dim ** -0.5
# #
# #         # 头部划分
# #         self.l_heads = max(0, int(num_heads * alpha))
# #         self.h_heads = num_heads - self.l_heads
# #
# #         self.l_dim = self.l_heads * head_dim
# #         self.h_dim = self.h_heads * head_dim
# #
# #         if self.ws == 1:
# #             # 全局模式退化为全局注意力
# #             self.h_heads = 0
# #             self.h_dim = 0
# #             self.l_heads = num_heads
# #             self.l_dim = dim
# #
# #         # 全局支路
# #         if self.l_heads > 0:
# #             if self.ws != 1:
# #                 self.sr = nn.AvgPool2d(kernel_size=self.ws, stride=self.ws)
# #             self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
# #             self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
# #             self.l_proj = nn.Linear(self.l_dim, self.l_dim)
# #
# #         # 局部支路
# #         if self.h_heads > 0:
# #             self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
# #             self.h_proj = nn.Linear(self.h_dim, self.h_dim)
# #
# #         self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
# #         self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()
# #
# #     def _loc(self, x):
# #         # x: [B, H, W, C]
# #         B, H, W, C = x.shape
# #         assert H % self.ws == 0 and W % self.ws == 0, "H/W must be divisible by window_size"
# #         h_group, w_group = H // self.ws, W // self.ws
# #         total_groups = h_group * w_group
# #
# #         # 先分块: [B, h_group, ws, w_group, ws, C] -> 交换 ws 维度 -> [B, h_group, w_group, ws, ws, C]
# #         x_blk = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
# #         # 变成 [B, groups, ws*ws, C]
# #         x_blk = x_blk.reshape(B, total_groups, self.ws * self.ws, C)
# #
# #         qkv = self.h_qkv(x_blk)  # [B, groups, ws*ws, 3*h_dim]
# #         qkv = qkv.reshape(B, total_groups, self.ws * self.ws, 3, self.h_heads, self.h_dim // self.h_heads)
# #         qkv = qkv.permute(3, 0, 1, 4, 2, 5)  # 3 x [B, groups, heads, tokens, dim]
# #         q, k, v = qkv[0], qkv[1], qkv[2]
# #         attn = (q @ k.transpose(-2, -1)) * self.scale
# #         attn = attn.softmax(dim=-1)
# #         attn = self.attn_drop(attn)
# #
# #         out = (attn @ v)  # [B, groups, heads, tokens, dim]
# #         out = out.transpose(2, 3).reshape(B, total_groups, self.ws * self.ws, self.h_dim)
# #         # fold 回 [B, H, W, C_h]
# #         out = out.reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim).transpose(2, 3)
# #         out = out.reshape(B, H, W, self.h_dim)
# #         out = self.h_proj(out)
# #         out = self.proj_drop(out)
# #         return out
# #
# #     def _glo(self, x):
# #         # x: [B, H, W, C]
# #         B, H, W, C = x.shape
# #         q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)
# #         if self.ws > 1:
# #             x_ = x.permute(0, 3, 1, 2)                   # [B, C, H, W]
# #             x_ = self.sr(x_)                             # [B, C, H/ws, W/ws]
# #             x_ = x_.reshape(B, C, -1).permute(0, 2, 1)   # [B, (H*W/ws^2), C]
# #             kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads)
# #         else:
# #             kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads)
# #         kv = kv.permute(2, 0, 3, 1, 4)
# #         k, v = kv[0], kv[1]  # [B, heads, tokens, dim]
# #
# #         attn = (q @ k.transpose(-2, -1)) * self.scale
# #         attn = attn.softmax(dim=-1)
# #         attn = self.attn_drop(attn)
# #
# #         out = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
# #         out = self.l_proj(out)
# #         out = self.proj_drop(out)
# #         return out
# #
# #     def forward(self, x, H, W):
# #         # x: [B, N, C] -> [B, H, W, C]
# #         B, N, C = x.shape
# #         assert N == H * W
# #         x = x.reshape(B, H, W, C)
# #
# #         if self.h_heads == 0:
# #             out = self._glo(x)
# #         elif self.l_heads == 0:
# #             out = self._loc(x)
# #         else:
# #             out = torch.cat([self._loc(x), self._glo(x)], dim=-1)  # [B, H, W, C_h+C_l] = [B, H, W, C]
# #
# #         return out.reshape(B, N, C)
# #
# #
# # # -----------------------------
# # # Cross-Gated Feed-Forward
# # # -----------------------------
# # class FeedForward(nn.Module):
# #     """
# #     Cross-Gated FFN，通道维做 1x1 -> DWConv -> 1x1；中间两路互相门控。
# #     保持 token 数不变：输入 [B, L, C]，输出 [B, L, C]
# #     """
# #     def __init__(self, dim: int, window_size: int, ffn_expansion_factor: float = 2.66, bias: bool = True):
# #         super().__init__()
# #         hidden = int(dim * ffn_expansion_factor)
# #         self.ws = window_size
# #         self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
# #         self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1,
# #                                 groups=hidden * 2, bias=bias)
# #         self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)
# #
# #     def forward(self, x):
# #         # x: [B, L, C] with L = ws*ws
# #         B, L, C = x.shape
# #         ws = self.ws
# #         assert L == ws * ws, "L must equal window_size*window_size for this FFN"
# #         x = x.view(B, ws, ws, C).permute(0, 3, 1, 2)    # [B, C, ws, ws]
# #
# #         x = self.project_in(x)
# #         x1, x2 = self.dwconv(x).chunk(2, dim=1)
# #         x = F.gelu(x2) * x1 + F.gelu(x1) * x2
# #         x = self.project_out(x)                         # [B, C, ws, ws]
# #
# #         x = x.permute(0, 2, 3, 1).reshape(B, L, C)      # [B, L, C]
# #         return x
# #
# #
# # # -----------------------------
# # # Transformer Block
# # # -----------------------------
# # class Block(nn.Module):
# #     def __init__(
# #         self,
# #         dim: int,
# #         num_heads: int,
# #         window_size: int = 3,
# #         mlp_ratio: float = 2.66,
# #         qkv_bias: bool = False,
# #         qk_scale=None,
# #         drop_ratio: float = 0.0,
# #         attn_drop_ratio: float = 0.0,
# #         drop_path_ratio: float = 0.0,
# #         norm_layer=nn.LayerNorm,
# #     ):
# #         super().__init__()
# #         self.ws = window_size
# #         self.norm1 = norm_layer(dim)
# #         self.attn = GLA(
# #             dim=dim,
# #             num_heads=num_heads,
# #             qkv_bias=qkv_bias,
# #             qk_scale=qk_scale,
# #             attn_drop=attn_drop_ratio,
# #             proj_drop=drop_ratio,
# #             window_size=window_size,
# #             alpha=0.5,
# #         )
# #         self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0.0 else nn.Identity()
# #         self.norm2 = norm_layer(dim)
# #         self.ffn = FeedForward(dim, window_size=window_size, ffn_expansion_factor=mlp_ratio)
# #
# #     def forward(self, x):
# #         # x: [B, L, C] , L should be (ws*ws) * (some integer), 这里上游保证 L = H*W
# #         B, L, C = x.shape
# #         H = W = int(math.sqrt(L))
# #         assert H * W == L, "Sequence length must be a perfect square (H*W)."
# #         x = x + self.drop_path(self.attn(self.norm1(x), H, W))
# #         x = x + self.drop_path(self.ffn(self.norm2(x)))
# #         return x
# #
# #
# # # -----------------------------
# # # GLAFormer (dual-branch, fuse by |x1 - x2|)
# # # -----------------------------
# # class GLAFormer(nn.Module):
# #     """
# #     - 两路输入（时相 T1 / T2），像素级嵌入 + 多层 Block
# #     - 融合策略：|feat1 - feat2|
# #     - Head: 小型 conv + fc 输出 2 类 logits
# #     期望每个样本是一个 window 的 patch；支持两种输入形式：
# #       (A) 序列： [B, L=ws*ws, C_in]
# #       (B) 图像： [B, C_in, ws, ws]
# #     """
# #     def __init__(
# #         self,
# #         img_c: int,
# #         num_classes: int = 2,
# #         window_size: int = 6,
# #         embed_dim: int = 256,
# #         depth: int = 6,
# #         num_heads: int = 8,
# #         mlp_ratio: float = 4.0,
# #         qkv_bias: bool = True,
# #         qk_scale=None,
# #         drop_ratio: float = 0.0,
# #         attn_drop_ratio: float = 0.0,
# #         drop_path_ratio: float = 0.0,
# #         block_window_size: int = 3,   # 注意力/FFN 里的局部窗口
# #     ):
# #         super().__init__()
# #         self.num_classes = num_classes
# #         self.embed_dim = embed_dim
# #         self.window_size = window_size
# #         L = window_size * window_size
# #
# #         # 两支嵌入
# #         self.img1_patch_embed = PixelEmbedding(num_bands=img_c, embed_dim=embed_dim, window_size=window_size)
# #         self.img2_patch_embed = PixelEmbedding(num_bands=img_c, embed_dim=embed_dim, window_size=window_size)
# #
# #         # 两支 Transformer
# #         dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
# #         self.img1_blocks = nn.Sequential(*[
# #             Block(dim=embed_dim, num_heads=num_heads, window_size=block_window_size,
# #                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
# #                   drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
# #                   drop_path_ratio=dpr[i], norm_layer=nn.LayerNorm)
# #             for i in range(depth)
# #         ])
# #         self.img1_norm = nn.LayerNorm(embed_dim)
# #
# #         self.img2_blocks = nn.Sequential(*[
# #             Block(dim=embed_dim, num_heads=num_heads, window_size=block_window_size,
# #                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
# #                   drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
# #                   drop_path_ratio=dpr[i], norm_layer=nn.LayerNorm)
# #             for i in range(depth)
# #         ])
# #         self.img2_norm = nn.LayerNorm(embed_dim)
# #
# #         # Head: conv -> conv -> fc
# #         self.conv1 = nn.Sequential(
# #             nn.Conv2d(in_channels=embed_dim, out_channels=16, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True)
# #         )
# #         self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
# #         self.fc = nn.Sequential(
# #             nn.Linear(L, 20),
# #             nn.ReLU(inplace=True),
# #             nn.Linear(20, num_classes)
# #         )
# #
# #         self.apply(_init_vit_weights)
# #
# #     # --------- 统一入口：自动识别输入形状 ----------
# #     def forward(self, img1, img2):
# #         """
# #         img1, img2:
# #           - 序列模式: [B, L=ws*ws, C_in]
# #           - 图像模式: [B, C_in, ws, ws]
# #         返回: [B, num_classes] logits
# #         """
# #         if img1.ndim == 4:
# #             # [B, C, ws, ws] -> [B, L, C]
# #             B, C_in, H, W = img1.shape
# #             assert H == self.window_size and W == self.window_size, "H/W must equal window_size"
# #             img1 = img1.permute(0, 2, 3, 1).reshape(B, H * W, C_in)
# #             img2 = img2.permute(0, 2, 3, 1).reshape(B, H * W, C_in)
# #         else:
# #             # [B, L, C]
# #             B, L, C_in = img1.shape
# #             assert L == self.window_size * self.window_size, "L must be window_size*window_size"
# #
# #         # 嵌入 + 编码
# #         x1 = self.img1_patch_embed(img1)
# #         x1 = self.img1_blocks(x1)
# #         x1 = self.img1_norm(x1)
# #
# #         x2 = self.img2_patch_embed(img2)
# #         x2 = self.img2_blocks(x2)
# #         x2 = self.img2_norm(x2)
# #
# #         # 融合：绝对差
# #         fuse = torch.abs(x1 - x2)  # [B, L, D]
# #
# #         # Head
# #         ws = self.window_size
# #         B, L, D = fuse.shape
# #         x = fuse.view(B, ws, ws, D).permute(0, 3, 1, 2)  # [B, D, ws, ws]
# #         x = self.conv1(x)
# #         x = self.conv2(x)                                # [B, 1, ws, ws]
# #         x = x.view(B, -1)                                # [B, L]
# #         logits = self.fc(x)                              # [B, num_classes]
# #         return logits
# #
# #
# #
# # Cross Gated Feed-Forward Network
# from functools import partial
#
# from einops import rearrange
# class PixelEmbedding(nn.Module):
#
#     def __init__(self, num_bands, embed_dim):
#         super(PixelEmbedding, self).__init__()
#         self.fc = nn.Linear(num_bands, embed_dim)
#         windowSize=9
#         self.positional_encoding = nn.Parameter(torch.randn(1, windowSize * windowSize, embed_dim))
#         nn.init.trunc_normal_(self.positional_encoding, std=0.02)
#
#     def forward(self, x):
#         # (batch_size, seq_len(9*9), num_bands)
#         if x.ndim == 4:
#             B, C, H, W = x.shape
#             # 转换为 [B, H*W, C] (例如 1x81x155)
#             x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
#
#         x = self.fc(x)
#         x += self.positional_encoding
#         return x  # [B , 81 ,384]
#
#
# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output
#
#
# class DropPath(nn.Module):
#
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)
#
# class GLA(nn.Module):
#
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=3, alpha=0.5):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#         head_dim = int(dim/num_heads)
#         self.dim = dim
#
#         self.l_heads = int(num_heads * alpha)
#         self.l_dim = self.l_heads * head_dim
#
#         self.h_heads = num_heads - self.l_heads
#         self.h_dim = self.h_heads * head_dim
#
#         self.ws = window_size
#
#         if self.ws == 1:
#             self.h_heads = 0
#             self.h_dim = 0
#             self.l_heads = num_heads
#             self.l_dim = dim
#
#         self.scale = qk_scale or head_dim ** -0.5
#
#         if self.l_heads > 0:
#             if self.ws != 1:
#                 self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
#             self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
#             self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
#             self.l_proj = nn.Linear(self.l_dim, self.l_dim)
#
#         if self.h_heads > 0:
#             self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
#             self.h_proj = nn.Linear(self.h_dim, self.h_dim)
#
#     def loc(self, x):
#         B, H, W, C = x.shape
#         h_group, w_group = H // self.ws, W // self.ws
#
#         total_groups = h_group * w_group
#
#         x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
#
#         qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
#         attn = attn.softmax(dim=-1)
#         attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
#         x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)
#
#         x = self.h_proj(x)
#         return x
#
#     def glo(self, x):
#         B, H, W, C = x.shape
#
#         q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)
#
#         if self.ws > 1:
#             x_ = x.permute(0, 3, 1, 2)
#             x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
#             kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
#         else:
#             kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
#         x = self.l_proj(x)
#         return x
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#
#         x = x.reshape(B, H, W, C)
#
#         if self.h_heads == 0:
#             x = self.glo(x)
#             return x.reshape(B, N, C)
#
#         if self.l_heads == 0:
#             x = self.loc(x)
#             return x.reshape(B, N, C)
#
#         loc_out = self.loc(x)
#         glo_out = self.glo(x)
#
#         x = torch.cat((loc_out, glo_out), dim=-1)
#         x = x.reshape(B, N, C)
#
#         return x
#
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor=2.66, bias=True):
#         super(FeedForward, self).__init__()
#         hidden_features = int(dim * ffn_expansion_factor)
#         self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
#         self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
#                                 groups=hidden_features * 2, bias=bias)
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         B, _, C = x.shape
#         # x = torch.permute(x.view(B,9,9,C),(0,3,1,2)) # [B,C,9,9]
#         windowSize=9
#         x = rearrange(x, 'b (h w) c -> b c h w', h=windowSize, w=windowSize)
#
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x2) * x1 + F.gelu(x1) * x2
#         x = self.project_out(x)
#
#         # x = torch.permute(x,(0,2,3,1)).view(B,81,C) # [B,9*9,C]
#         x = rearrange(x, 'b c h w -> b (h w) c')
#         return x
#
#
# class Block(nn.Module):
#     def __init__(self,
#                  dim,
#                  num_heads,
#                  mlp_ratio=2.66,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  drop_ratio=0.,
#                  attn_drop_ratio=0.,
#                  drop_path_ratio=0.,
#                  act_layer=nn.GELU,
#                  norm_layer=nn.LayerNorm):
#         super(Block, self).__init__()
#         self.norm1 = norm_layer(dim)
#         # self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#         #                       attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
#
#         self.gla = GLA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                        attn_drop=attn_drop_ratio, proj_drop=drop_ratio, window_size=3, alpha=0.5)
#
#         self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         self.ffn = FeedForward(dim, ffn_expansion_factor=mlp_ratio)
#         # mlp_hidden_dim = int(dim * mlp_ratio)
#         # self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
#
#     def forward(self, x):
#         windowSize=9
#         x = x + self.drop_path(self.gla(self.norm1(x), windowSize, windowSize))  # norm1-->attn-->drop_path
#         x = x + self.drop_path(self.ffn(self.norm2(x)))  # norm2-->MLP(FFN)-->drop_path
#         return x
# def _init_vit_weights(m):
#     """
#     ViT weight initialization
#     :param m: module
#     """
#     if isinstance(m, nn.Linear):
#         nn.init.trunc_normal_(m.weight, std=.01)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
#     elif isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode="fan_out")
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
#     elif isinstance(m, nn.LayerNorm):
#         nn.init.zeros_(m.bias)
#         nn.init.ones_(m.weight)
# class GLAFormer(nn.Module):
#     def __init__(self,  img_c, num_classes=2,
#                  embed_dim=128, depth=4, num_heads=8, mlp_ratio=4.0, qkv_bias=True,
#                  qk_scale=None, drop_ratio=0.,
#                  attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PixelEmbedding, norm_layer=None,
#                  act_layer=None):
#
#         super(GLAFormer, self).__init__()
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         self.num_tokens = 1
#         img1_norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#         img1_act_layer = act_layer or nn.GELU
#         img2_norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#         img2_act_layer = act_layer or nn.GELU
#
#         self.img1_patch_embed = embed_layer(num_bands = img_c, embed_dim = embed_dim)
#         self.img2_patch_embed = embed_layer(num_bands = img_c, embed_dim = embed_dim)
#
#         img1_dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
#         self.img1_blocks = nn.Sequential(*[
#             Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                   drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=img1_dpr[i],
#                   norm_layer=img1_norm_layer, act_layer=img1_act_layer)
#             for i in range(depth)
#         ])
#         self.img1_norm = img1_norm_layer(embed_dim)
#
#         img2_dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
#         self.img2_blocks = nn.Sequential(*[
#             Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                   drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=img1_dpr[i],
#                   norm_layer=img2_norm_layer, act_layer=img2_act_layer)
#             for i in range(depth)
#         ])
#         self.img2_norm = img2_norm_layer(embed_dim)
#         windowSize=9
#         # Classifier head(s)
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels=embed_dim, out_channels=16, kernel_size=3, padding=1),nn.ReLU())
#         self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1))
#         self.fc = nn.Sequential(
#             nn.Linear(windowSize*windowSize, 20),
#             nn.ReLU(),
#             nn.Linear(20, 2)
#         )
#
#         self.apply(_init_vit_weights)
#
#     def forward_features(self, img):
#         x = self.patch_embed(x , y)
#         x = self.blocks(x)
#         x = self.norm(x)
#         return x
#
#     def forward(self, img1, img2):
#         windowSize=9
#         img1 = self.img1_patch_embed(img1)
#         img1 = self.img1_blocks(img1)
#         img1 = self.img1_norm(img1)
#
#
#         img2 = self.img1_patch_embed(img2)
#         img2 = self.img1_blocks(img2)
#         img2 = self.img1_norm(img2)
#
#         fuse_feature = torch.abs(torch.sub(img1, img2))
#         # print(fuse_feature.shape)
#         B,_ , C = fuse_feature.shape
#         x = torch.torch.permute(fuse_feature.view(B,windowSize,windowSize,C),(0,3,1,2))
#         # print(x.shape)
#         x = self.conv1(x)
#         x = self.conv2(x)
#
#         x = x.view(B,-1)
#         x = self.fc(x)
#
#         return x
#
# import torch
#
# # 如果你要统计 FLOPs / Params，安装 thop: pip install thop
# try:
#     from thop import profile
# except ImportError:
#     profile = None
#
#
# if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     # 你的实现写死 windowSize=9 => L=81
#     B = 1
#     windowSize = 9
#     L = windowSize * windowSize
#     C_in = 155
#     num_classes = 2
#
#     model = GLAFormer(
#         img_c=C_in,
#         num_classes=num_classes,
#         embed_dim=256,
#         depth=6,
#         num_heads=8,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         drop_ratio=0.0,
#         attn_drop_ratio=0.0,
#         drop_path_ratio=0.0
#     ).to(device)
#
#     model.eval()
#
#     # ✅ PixelEmbedding 需要 (B, 81, 155)
#     x1 = torch.randn(B, L, C_in, device=device)
#     x2 = torch.randn(B, L, C_in, device=device)
#
#     print(f"输入形状: x1={x1.shape}, x2={x2.shape}")
#
#     # 参数量
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"总参数量: {total_params:,}")
#     print(f"可训练参数量: {trainable_params:,}")
#
#     # 前向推理
#     with torch.no_grad():
#         out = model(x1, x2)
#
#     print(f"输出形状: out={out.shape}")
#     print(f"输出示例: {out[:2]}")
#
#     # FLOPs / MACs（可选）
#     if profile is not None:
#         macs, params = profile(model, inputs=(x1, x2), verbose=False)
#         print(f"计算量 MACs：{macs/1e6:.4f} M")
#         print(f"计算量 FLOPs：{2*macs/1e6:.4f} M")   # 1 MAC ≈ 2 FLOPs
#         print(f"参数量 Params：{params/1e3:.4f} K")
#     else:
#         print("未安装 thop，跳过 MACs/FLOPs 统计。可执行: pip install thop")
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