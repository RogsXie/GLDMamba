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
