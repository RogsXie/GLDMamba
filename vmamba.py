import math
from functools import partial
from typing import Any
from collections import OrderedDict
from typing import Optional, Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except:
    from csm_triton import cross_scan_fn, cross_merge_fn

try:
    from .csms6s import selective_scan_fn
except:
    from csms6s import selective_scan_fn

class _ChannelAttn(nn.Module):
    def __init__(self, channels, reduction=16, channel_first=True):
        super().__init__()
        self.channel_first = channel_first
        mid = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        out = x * w
        if not self.channel_first:
            out = out.permute(0, 2, 3, 1).contiguous()
        return out

class _SpatialAttn(nn.Module):
    def __init__(self, kernel_size=7, channel_first=True):
        super().__init__()
        self.channel_first = channel_first
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        nhwc = False
        if not self.channel_first:
            nhwc = True
            x = x.permute(0, 3, 1, 2).contiguous()
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg, mx], dim=1)
        w = torch.sigmoid(self.conv(a))
        out = x * w
        if nhwc:
            out = out.permute(0, 2, 3, 1).contiguous()
        return out

class CBAM2d(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7, channel_first=True):
        super().__init__()
        self.ca = _ChannelAttn(channels, reduction=reduction, channel_first=channel_first)
        self.sa = _SpatialAttn(kernel_size=kernel_size, channel_first=channel_first)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                             missing_keys, unexpected_keys, error_msgs)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError

# -------------------- Mamba params init --------------------
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)\
                  .view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init,
                    dt_min, dt_max, dt_init_floor, k_group=4):
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in dt_projs], dim=0))   # (K, inner, rank)
        dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in dt_projs], dim=0))     # (K, inner)
        del dt_projs
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        Ds     = cls.D_init(d_inner, copies=k_group, merge=True)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias

class SS2Dv2:
    def __initv2__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v2",
            channel_first=False,
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model  = int(d_model)
        self.d_state  = int(d_state)
        self.d_inner  = int(ssm_ratio * d_model)
        self.dt_rank  = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv    = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        self.use_cbam   = kwargs.get("use_cbam", True)
        self.cbam_on    = kwargs.get("cbam_on", "z")
        self.cbam_red   = kwargs.get("cbam_reduction", 16)
        self.cbam_ks    = kwargs.get("cbam_kernel", 7)
        self.use_gate_split = kwargs.get("use_gate_split", False)
        if self.use_cbam:
            self.cbam = CBAM2d(self.d_inner, reduction=self.cbam_red,
                               kernel_size=self.cbam_ks,
                               channel_first=self.channel_first)

        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32",     forward_type)
        self.oact,            forward_type = checkpostfix("_oact",     forward_type)
        self.disable_z,       forward_type = checkpostfix("_noz",      forward_type)
        self.disable_z_act,   forward_type = checkpostfix("_nozact",   forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        FORWARD_TYPES = dict(
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                        selective_scan_backend="mamba"),
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            v2 =partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                        selective_scan_backend="core"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES["v2"])

        # in proj
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # depthwise conv
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0))   # (K, N, inner)
        del self.x_proj

        # out proj
        self.out_act  = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout  = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # params init
        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = \
                mamba_init.init_dt_A_D(
                    self.d_state, self.dt_rank, self.d_inner,
                    dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                    k_group=self.k_group,
                )
        else:
            self.Ds              = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs          = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias   = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

        # ==================================================================
        #  新增参数：可学习漂移感知网络（Learnable Drift-Aware SSM）
        #
        #  核心升级：将 v 的估计从手工统计量（均值/std）改为可学习网络。
        #
        #  原版 v 的问题：
        #    v = sigmoid(|x - mean(x)| / std(x))
        #    这是一个固定的统计公式，梯度被 no_grad 截断，
        #    drift_dt_scale 和 drift_A_bias 只能通过下游损失间接学习，
        #    且统计量无法区分"真正的变化"与"正常的特征多样性"。
        #
        #  新方案：用 drift_score_net 直接从特征预测逐位置漂移概率，
        #    梯度完全流通，模型可以端到端学习"什么样的特征模式对应变化"。
        #
        #  (1) drift_score_net：[B, D, L] -> [B, K, L]
        #      输入：cross_scan 展开前的原始特征 x [B, D, H, W]
        #      结构：DW3x3 → PW → ReLU → PW → Sigmoid
        #      输出：K 个方向各自的逐像素漂移概率图，∈ (0,1)
        #      初始化偏置为负值 → 初始输出接近 0 → 训练初期退化为原始 SSM
        #
        #  (2) drift_dt_scale [K, D]：各方向各通道的步长敏感度
        #      Δ̃_{k,i} = Δ_{k,i} · (1 + drift_dt_scale[k] · v_{k,i})
        #
        #  (3) drift_A_bias [K, D, N]：各方向各通道的衰减补偿
        #      Ã_log[k,d] = A_log[k,d] - |drift_A_bias[k,d]| · v_mean[k,d]
        # ==================================================================
        inter = max(self.d_inner // 4, 16)   # 瓶颈维度
        self.drift_score_net = nn.Sequential(
            # PW 降维：逐通道点级映射，无局部感受野
            nn.Conv2d(self.d_inner, inter, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            # PW 升维到 K 个方向的漂移概率图
            nn.Conv2d(inter, self.k_group, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        # 初始化最后一层偏置为负值，使初始输出 ≈ sigmoid(-2) ≈ 0.12 ≈ 0
        nn.init.constant_(self.drift_score_net[-2].bias, -2.0)

        self.drift_dt_scale = nn.Parameter(
            torch.zeros(self.k_group, self.d_inner)                # [K, D]
        )
        self.drift_dt_scale._no_weight_decay = True

        self.drift_A_bias = nn.Parameter(
            torch.zeros(self.k_group, self.d_inner, self.d_state)  # [K, D, N]
        )
        self.drift_A_bias._no_weight_decay = True

    # ------------------------------------------------------------------
    #  forward_corev2  ——  唯一实质改动处
    # ------------------------------------------------------------------
    def forward_corev2(
            self,
            x: torch.Tensor = None,
            force_fp32=False,
            ssoflex=True,
            no_einsum=False,
            selective_scan_backend=None,
            scan_mode="cross2d",
            scan_force_torch=False,
            **kwargs,
    ):
        """
        与原版接口完全一致（签名不变）。

        内部新增两步（均在 selective_scan 调用之前）：

        Step A — 估计局部漂移强度 v [B, K*D, L]
            利用特征图 x 自身：每个位置与空间均值的归一化绝对偏差，
            作为"该位置发生时相漂移的程度"的代理指标。
            公式：v_i = |x_i - mean_spatial(x)| / (std_spatial(x) + ε)
                   v_i = sigmoid(v_i)   ∈ (0, 1)
            无需外部输入，接口零改动。

        Step B1 — 漂移感知 Δt 调制
            Δ̃ = Δ · (1 + drift_dt_scale · v)
            变化区域步长更大 → SSM 对光谱跳变响应更快。

        Step B2 — 漂移感知 A 补偿
            Ã_log = A_log - |drift_A_bias| · mean(v, dim=L)
            变化区域 |A| 减小 → 状态衰减更慢 → 历史上下文保留更久，
            有助于区分真实变化与短暂扰动。
        """
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, 0)
        delta_softplus = True
        out_norm     = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None,
                           delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias,
                                     delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        # ---------- 标准扫描 & x_proj（与原版完全一致）----------
        x_proj_bias = getattr(self, "x_proj_bias", None)
        xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True,
                           scans=_scan_mode, force_torch=scan_force_torch)

        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L),
                             self.x_proj_weight.view(-1, D, 1),
                             bias=(x_proj_bias.view(-1)
                                   if x_proj_bias is not None else None),
                             groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = F.conv1d(dts.contiguous().view(B, -1, L),
                               self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs  = xs.view(B, -1, L)                         # [B, K*D, L]
        dts = dts.contiguous().view(B, -1, L)            # [B, K*D, L]

        # ---------- 原版 A / D / delta_bias ----------
        As          = -self.A_logs.to(torch.float32).exp()   # [K*D, N]
        Ds          = self.Ds.to(torch.float32)               # [K*D]
        Bs          = Bs.contiguous().view(B, K, N, L)
        Cs          = Cs.contiguous().view(B, K, N, L)
        delta_bias  = self.dt_projs_bias.view(-1).to(torch.float32)  # [K*D]

        # ==================================================================
        #  Step A：可学习漂移概率图 v  [B, K, D, L]
        #
        #  用 drift_score_net 直接从输入特征图 x [B, D, H, W] 预测：
        #    score [B, K, H, W] = DW3x3 → PW → ReLU → PW → Sigmoid
        #
        #  相比手工统计量的三点优势：
        #    1. 梯度完全流通，end-to-end 学习"什么特征模式对应变化"
        #    2. DW3x3 捕获局部空间结构，而非单纯的全局统计
        #    3. K 个输出通道对应 K 个扫描方向，各自独立预测
        #
        #  score: [B, K, H, W] → reshape [B, K, 1, L] → 广播到 [B, K, D, L]
        #  注：v 在 D 维度上广播（同一位置所有通道共享漂移概率），
        #      具体的通道差异由 drift_dt_scale [K, D] 控制。
        # ==================================================================
        score = self.drift_score_net(x)                         # [B, K, H, W]
        # [B,K,H,W] -> [B,K,1,L]，广播到 [B,K,D,L]，contiguous保证梯度正常回传
        v = score.view(B, K, 1, L).expand(B, K, D, L).contiguous()  # [B, K, D, L]

        # ==================================================================
        #  Step B1：方向独立的 Δt 调制
        #
        #  修改：Δ̃_{k,i} = Δ_{k,i} · (1 + drift_dt_scale[k] · v_{k,i})
        #
        #  drift_dt_scale [K, D]：第 k 方向、第 d 通道各自学习敏感度，
        #  不同扫描方向对同一变化区域的响应强度可以不同。
        #  初始化为 0 → 训练初期退化为原始 SSM。
        # ==================================================================
        scale = self.drift_dt_scale.to(torch.float32)          # [K, D]
        # scale: [K,D] -> [1,K,D,1]，v: [B,K,D,L] → 逐方向逐通道调制
        dts_kd = dts.view(B, K, D, L)
        dts_kd = dts_kd * (1.0 + scale.unsqueeze(0).unsqueeze(-1) * v)
        dts = dts_kd.view(B, K * D, L)                         # 还原 [B, K*D, L]

        # ==================================================================
        #  Step B2：方向独立的 A 补偿
        #
        #  修改：Ã_log[k,d] = A_log[k,d] - |drift_A_bias[k,d]| · ṽ[k,d]
        #        ṽ[k,d] = mean_{B,L}(v[k,d])  ∈ (0,1)
        #
        #  drift_A_bias [K, D, N]：第 k 方向独立学习衰减补偿，
        #  水平/垂直扫描路径对变化区域的状态记忆需求不同，各自自适应。
        #  cuda kernel 要求 A 严格为 [K*D, N]，最后 reshape 回去。
        # ==================================================================
        # v: [B,K,D,L] → 折叠 B 和 L → [K, D]
        v_mean = v.mean(dim=-1).mean(dim=0)                     # [K, D]
        A_bias = self.drift_A_bias.to(torch.float32).abs()      # [K, D, N]
        # v_mean [K,D] x A_bias [K,D,N] → [K,D,N]，按方向逐元素相乘
        A_drift = (v_mean.unsqueeze(-1) * A_bias)               # [K, D, N]
        # As [K*D, N] reshape → [K,D,N]，减去各方向补偿后还原
        As = (As.view(K, D, N) - A_drift).view(K * D, N)        # [K*D, N]
        # ==================================================================

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        # ---------- selective_scan（形状与原版完全一致）----------
        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)

        y: torch.Tensor = cross_merge_fn(
            ys, in_channel_first=True, out_channel_first=True,
            scans=_scan_mode, force_torch=scan_force_torch
        )

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(1, 2).contiguous().view(B, H, W, -1)
        y = out_norm(y)
        return y.to(x.dtype)

    # ------------------------------------------------------------------
    #  以下全部与原版一致，零改动
    # ------------------------------------------------------------------
    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)

        if getattr(self, "use_cbam", False):
            if self.cbam_on in ("y", "yz"):
                y = self.cbam(y)
            if (self.cbam_on in ("z", "yz")) and (not self.disable_z):
                z = self.cbam(z)

        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value.endswith(tag)
            return (ret, value[:-len(tag)] if ret else value)

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        out_norm_none,    forward_type = checkpostfix("_onnone",    forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm,   forward_type = checkpostfix("_oncnorm",   forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)
        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value.endswith(tag)
        return (ret, value[:-len(tag)] if ret else value)

class SS2D(nn.Module, SS2Dv2):
    def __init__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v2",
            channel_first=False,
            **kwargs,
    ):
        nn.Module.__init__(self)
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale,
            dt_init_floor=dt_init_floor, initialize=initialize,
            forward_type=forward_type, channel_first=channel_first,
        )
        self.__initv2__(**kwargs)

class VSSBlock(nn.Module):
    def __init__(
            self,
            dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            dt_rank: Any = "auto",
            ssm_ratio=2.0,
            shared_ssm=False,
            softmax_version=False,
            use_checkpoint: bool = False,
            mlp_ratio=4.0,
            act_layer=nn.GELU,
            drop: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(dim)
        self.op = SS2D(
            d_model=dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           channels_first=False)

    def _forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.op(self.norm(input.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)  # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

class VSSM(nn.Module):
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=2.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",
            patchembed_version: str = "v1",
            use_checkpoint=False,
            posembed=False,
            imgsize=224,
            _SS2D=SS2D,
            **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes  = num_classes
        self.num_layers   = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        _NORMLAYERS = dict(ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS  = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)

        norm_layer:    nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(
            in_chans, dims[0], patch_size, patch_norm, norm_layer,
            channel_first=self.channel_first)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                channel_first=self.channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                _SS2D=_SS2D,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features),
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True,
                          norm_layer=nn.LayerNorm, channel_first=False):
        padding = (patch_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                      stride=1, padding=padding, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True,
                              norm_layer=nn.LayerNorm, channel_first=False):
        stride      = patch_size // 2
        kernel_size = stride + 1
        padding     = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            channel_first=False,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            _SS2D=SS2D,
            **kwargs,
    ):
        depth  = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS2D=_SS2D,
            ))
        return nn.Sequential(OrderedDict(blocks=nn.Sequential(*blocks)))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = (self.pos_embed.permute(0, 2, 3, 1)
                         if not self.channel_first else self.pos_embed)
            x = x + pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x



