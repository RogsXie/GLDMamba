# import torch
# import torch.nn as nn
# import mamba_ssm
# from mamba_ssm import Mamba  # 导入官方 Mamba 模块
#
# # --- 1. Spectral Attention (光谱注意力) ---
# class SpectralAttention(nn.Module):
#     def __init__(self, channels):
#         super(SpectralAttention, self).__init__()
#         # 使用两层 MLP (Encoder-Decoder 结构) 增强光谱维度依赖
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // 2, channels),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         # x: [B, C, H, W]
#         b, c, h, w = x.size()
#         avg_pool = torch.mean(x, dim=(2, 3))
#         weight = self.fc(avg_pool).view(b, c, 1, 1)
#         return x * weight
#
#
# # --- 2. Spatial Attention (空间注意力) ---
# class SpatialAttention(nn.Module):
#     def __init__(self, channels):
#         super(SpatialAttention, self).__init__()
#         # 使用卷积层进行空间融合，不使用池化以避免信息丢失
#         self.conv = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         weight = self.conv(x)
#         return x * weight
#
#
# # --- 3. GASSM Patch 核心网络 ---
# class GASSM_PatchNet(nn.Module):
#     def __init__(self, in_channels, num_classes=2, patch_size=7):
#         super(GASSM_PatchNet, self).__init__()
#
#         # A. Global Attention Mechanism (GAM)
#         self.spectral_att = SpectralAttention(in_channels)
#         self.spatial_att = SpatialAttention(in_channels)
#
#         # B. Feature Extraction Module (特征提取与降维)
#         # 使用 1x1 卷积减少通道 (降维至 128)，2D 卷积提取空间特征 (降维至 64)
#         self.feature_extract = nn.Sequential(
#             nn.Conv2d(in_channels, 128, kernel_size=1),
#             nn.Conv2d(128, 64, kernel_size=5, padding=0)  # 7x7 经过 5x5(pad=0) 变为 3x3[cite: 1]
#         )
#
#         # C. SSM-based Mamba Block[cite: 1]
#         # d_model 设置为 64 (由上一层卷积核数量决定)[cite: 1]
#         self.mamba = Mamba(
#             d_model=64,  # 模型维度[cite: 1]
#             d_state=16,  # SSM 状态维度
#             d_conv=4,  # 局部卷积维度
#             expand=2  # 扩展因子
#         )
#
#         # D. Binary Decision Making (决策层)[cite: 1]
#         # 经过 Mamba 后特征图大小仍为 3x3, 通道为 64[cite: 1]
#         self.flatten = nn.Flatten()
#         self.classifier = nn.Linear(64 * 3 * 3, num_classes)
#
#     def forward(self, patch1, patch2):
#         # 1. 输入处理：计算二时相 Patch 的绝对差异[cite: 1]
#         x = torch.abs(patch1 - patch2)  # [B, BANDS, 7, 7]
#
#         # 2. 全球注意力提取[cite: 1]
#         x = self.spectral_att(x)
#         x = self.spatial_att(x)
#
#         # 3. 特征降维与空间提取[cite: 1]
#         # 输出尺寸变为 [B, 64, 3, 3][cite: 1]
#         f3 = self.feature_extract(x)
#
#         # 4. Mamba 序列建模[cite: 1]
#         b, c, h, w = f3.size()
#         # 将 2D 特征转为序列形式: [Batch, Sequence_Length, Channels] -> [B, 9, 64][cite: 1]
#         mamba_in = f3.view(b, c, -1).transpose(1, 2)
#         mamba_out = self.mamba(mamba_in)  # [B, 9, 64]
#
#         # 5. 二元决策分类[cite: 1]
#         feat = self.flatten(mamba_out)  # [B, 576]
#         logits = self.classifier(feat)  # [B, num_classes]
#         return logits
#
#
# # --- 测试运行 ---
# # --- 修改后的测试运行部分 ---
# if __name__ == "__main__":
#     # 1. 检查是否有可用的 GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"当前使用的设备: {device}")
#
#     if device.type == 'cpu':
#         print("警告: Mamba 算子通常需要 GPU 才能运行。如果只有 CPU，可能会继续报错。")
#
#     # 2. 模拟输入并转移到 GPU
#     # 必须使用 .to(device)
#     p1 = torch.randn(8, 198, 7, 7).to(device)
#     p2 = torch.randn(8, 198, 7, 7).to(device)
#
#     # 3. 实例化模型并转移到 GPU
#     model = GASSM_PatchNet(in_channels=198, num_classes=2).to(device)
#
#     # 4. 设置为评估模式（推荐）
#     model.eval()
#
#     # 5. 前向传播
#     with torch.no_grad():
#         output = model(p1, p2)
#
#     print(f"输入 Patch 尺寸: {p1.shape}")
#     print(f"输出分类概率尺寸: {output.shape}")
#     print(f"结果设备: {output.device}")
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from thop import profile

# --- 1. Spectral Attention (光谱注意力) ---
class SpectralAttention(nn.Module):
    def __init__(self, channels):
        super(SpectralAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = torch.mean(x, dim=(2, 3))
        weight = self.fc(avg_pool).view(b, c, 1, 1)
        return x * weight


# --- 2. Spatial Attention (空间注意力) ---
class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        # 修正：去掉第二个 BN 后多余的 ReLU，直接用 Sigmoid 输出注意力权重
        # 论文 GAM 结构：Conv→BN→ReLU→Conv→BN→Sigmoid
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()  # 修正：去掉了原来在此之前的 ReLU
        )

    def forward(self, x):
        weight = self.conv(x)
        return x * weight


# --- 3. GASSM Patch 核心网络 ---
class GASSM_PatchNet(nn.Module):
    def __init__(self, in_channels, num_classes=2, patch_size=7):
        super(GASSM_PatchNet, self).__init__()

        # A. Global Attention Mechanism (GAM)
        self.spectral_att = SpectralAttention(in_channels)
        self.spatial_att = SpatialAttention(in_channels)

        # B. Feature Extraction Module
        # 修正：两层卷积之间加入 BN + ReLU，符合论文特征提取模块的标准设计
        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),           # 修正：新增 BN
            nn.ReLU(inplace=True),         # 修正：新增 ReLU
            nn.Conv2d(128, 64, kernel_size=5, padding=0),  # 7×7 → 3×3
            nn.BatchNorm2d(64),            # 修正：新增 BN
            nn.ReLU(inplace=True)          # 修正：新增 ReLU
        )

        # C. SSM-based Mamba Block
        self.mamba = Mamba(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2
        )

        # D. Binary Decision Making
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(64 * 3 * 3, num_classes)

    def forward(self, x):
        # 修正：forward 接收已计算好的差分 patch（符合论文 Fig.2 输入处理逻辑）
        # x: [B, BANDS, 7, 7]，即 Td 的 patch

        # 1. 全局注意力
        x = self.spectral_att(x)
        x = self.spatial_att(x)

        # 2. 特征降维与空间提取，输出 [B, 64, 3, 3]
        f3 = self.feature_extract(x)

        # 3. Mamba 序列建模：[B, 64, 9] → [B, 9, 64]
        b, c, h, w = f3.size()
        mamba_in = f3.view(b, c, -1).transpose(1, 2)
        mamba_out = self.mamba(mamba_in)  # [B, 9, 64]

        # 4. 二元决策分类
        feat = self.flatten(mamba_out)    # [B, 576]
        logits = self.classifier(feat)    # [B, num_classes]
        return logits


# --- 测试运行 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")

    # 修正：模拟输入为差分 patch（而非两个独立 patch）
    diff_patch = torch.randn(1, 242, 7, 7).to(device)

    model = GASSM_PatchNet(in_channels=242, num_classes=2).to(device)
    model.eval()

    print(f"输入形状: x1={diff_patch.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # macs, params = profile(model, inputs=(diff_patch ), verbose=False)
    macs, params = profile(model, inputs=(diff_patch,), verbose=False)

    print(f"计算量 MACs：{macs / 1e6:.4f} M")
    print(f"计算量 FLOPs：{2 * macs / 1e6:.4f} M")  # 一个 MAC ≈ 2 FLOPs
    print(f"参数量 Params：{params / 1e3:.4f} K")

    print("\n开始前向传播测试...")
    model.eval()
    with torch.no_grad():
        out = model(diff_patch )
    print(f"最终输出形状: {out.shape}")
    # print(f"输出应该是: ({height * width}, {num_classes})")

    import time

    print("\n开始性能测试...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    iters = 10
    with torch.no_grad():
        for _ in range(iters):
            _ = model(diff_patch )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    avg = (time.time() - start_time) / iters
    print(f"平均前向传播时间: {avg * 1000:.2f} ms")
    print(f"FPS: {1.0 / avg:.2f}")

    print("\n✅ 四路结构测试完成！")

    # with torch.no_grad():
    #     y = net(x1, x2)
    # print("Input :", x1.shape, x2.shape)
    # print("Output:", y.shape)  # -> torch.Size([32, 2])
