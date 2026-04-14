
from thop import profile, clever_format
import torch.autograd.profiler as profiler
import os
import time
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy.io as sio
from sklearn import preprocessing, metrics
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def measure_forward_backward_flops(model, C, ws, device):
    """测量一次 forward FLOPs、backward FLOPs、参数量"""
    model.eval()

    # 构造 dummy 输入
    x1 = torch.randn(1, C, ws, ws).to(device)
    x2 = torch.randn(1, C, ws, ws).to(device)
    y = torch.randint(0, 2, (1,)).to(device)

    # ---- 测 Forward FLOPs ----
    flops_fwd, params = profile(model, inputs=(x1, x2))

    # ---- 测 Backward FLOPs ----
    criterion = nn.CrossEntropyLoss()
    with profiler.profile(use_cuda=True) as prof:
        logits = model(x1, x2)
        loss = criterion(logits, y)
        loss.backward()

    total_backward_ops = sum([evt.cpu_time_total for evt in prof.key_averages()])

    return flops_fwd, total_backward_ops, params


import modelpa  # 你的模型定义
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

SEED = 42
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
torch.cuda.empty_cache()



windowSize = 5
epochs     = 100
lr         = 1e-4
gamma      = 0.9
batch_size = 64
dataset_name = "farm_patch"
save_dir_results = "results"
save_dir_models  = "model"
os.makedirs(save_dir_results, exist_ok=True)
os.makedirs(save_dir_models, exist_ok=True)

def pad_with_reflection(arr: np.ndarray, pad: int) -> np.ndarray:
    """ 边界反射填充，便于随处裁 patch；arr: (H, W, C) """
    if pad <= 0:
        return arr
    return np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

def linear_to_hw(idx: int, W: int):
    r = idx // W
    c = idx % W
    return int(r), int(c)

def draw_classification_map(label_2d: np.ndarray, save_path: str, scale: float = 4.0, dpi: int = 400):
    """ 保存灰度/索引图（用于简洁可视化） """
    fig, ax = plt.subplots()
    ax.imshow(label_2d.astype(np.int16), cmap='gray')
    ax.set_axis_off()
    fig.set_size_inches(label_2d.shape[1] * scale / dpi, label_2d.shape[0] * scale / dpi)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    fig.savefig(save_path, format='png', transparent=True, dpi=dpi, pad_inches=0)
    plt.close(fig)

def compute_metrics_arrays(y_true, y_pred, class_count: int):
    """
    y_true, y_pred: 1D numpy arrays, 取自非背景像素（>0）的标签（真实1..K；预测1..K）
    返回：OA、AA、每类精确率/召回/F1（长度K）、MacroP/R/F1、Kappa
    """
    eps = 1e-12
    # OA
    OA = (y_true == y_pred).mean() if y_true.size > 0 else 0.0

    y_true0 = y_true - 1
    y_pred0 = y_pred - 1

    TP = np.zeros(class_count, dtype=np.int64)
    FP = np.zeros(class_count, dtype=np.int64)
    FN = np.zeros(class_count, dtype=np.int64)
    SUP = np.zeros(class_count, dtype=np.int64)

    for k in range(class_count):
        mask_k = (y_true0 == k)
        SUP[k] = mask_k.sum()
        TP[k]  = np.sum(mask_k & (y_pred0 == k))
        FN[k]  = np.sum(mask_k & (y_pred0 != k))
        FP[k]  = np.sum((y_true0 != k) & (y_pred0 == k))

    acc_per_class = TP / (SUP + eps)
    valid_mask = SUP > 0
    AA = acc_per_class[valid_mask].mean() if valid_mask.any() else 0.0

    # P/R/F1
    P_c = TP / (TP + FP + eps)
    R_c = TP / (TP + FN + eps)
    F1_c = 2 * P_c * R_c / (P_c + R_c + eps)
    macro_P = P_c[valid_mask].mean() if valid_mask.any() else 0.0
    macro_R = R_c[valid_mask].mean() if valid_mask.any() else 0.0
    macro_F1 = F1_c[valid_mask].mean() if valid_mask.any() else 0.0

    # Kappa（用 sklearn，标签用 1..K）
    try:
        kappa = metrics.cohen_kappa_score(y_true.astype(np.int16), y_pred.astype(np.int16))
    except Exception:
        kappa = 0.0

    return {
        "OA": float(OA),
        "AA": float(AA),
        "per_class_acc": acc_per_class,
        "per_class_P": P_c,
        "per_class_R": R_c,
        "per_class_F1": F1_c,
        "macro_P": float(macro_P),
        "macro_R": float(macro_R),
        "macro_F1": float(macro_F1),
        "Kappa": float(kappa),
    }


class ChangePatchDataset(Dataset):

    def __init__(self, data1, data2, gt, index_list, ws=9):
        assert data1.shape == data2.shape
        self.H, self.W, self.C = data1.shape
        self.ws = ws
        self.pad = ws // 2

        self.d1p = pad_with_reflection(data1, self.pad)
        self.d2p = pad_with_reflection(data2, self.pad)
        # gt 保留 0..K（背景=0），取中心像素
        self.gtp = np.pad(gt, ((self.pad, self.pad), (self.pad, self.pad)),
                          mode='constant', constant_values=0)

        # 过滤仅保留前景索引
        fg = [int(i) for i in index_list if gt[linear_to_hw(int(i), self.W)] != 0]
        self.idxs = np.array(fg, dtype=np.int64)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        idx = int(self.idxs[i])
        r, c = linear_to_hw(idx, self.W)
        rp = r + self.pad
        cp = c + self.pad

        r0, r1 = rp - self.pad, rp + self.pad + 1
        c0, c1 = cp - self.pad, cp + self.pad + 1

        p1 = self.d1p[r0:r1, c0:c1, :]  # (ws, ws, C)
        p2 = self.d2p[r0:r1, c0:c1, :]
        y  = int(self.gtp[rp, cp])      # 原标签: 1..K

        y = y - 1

        # 转为 (C, ws, ws)
        p1 = np.transpose(p1, (2, 0, 1)).astype(np.float32)
        p2 = np.transpose(p2, (2, 0, 1)).astype(np.float32)

        x1 = torch.from_numpy(p1)
        x2 = torch.from_numpy(p2)
        y  = torch.tensor(y, dtype=torch.long)
        return x1, x2, y

@torch.no_grad()
def evaluate_val(model, loader, device, criterion, ws):
    model.eval()
    total_loss = 0.0
    correct = 0
    total   = 0
    n_samps = 0

    for x1, x2, y in loader:
        x1 = x1.to(device)  # [B, C, ws, ws]
        x2 = x2.to(device)
        y  = y.to(device)

        logits = model(x1, x2)              # (B, K)
        loss   = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total   += y.numel()
        n_samps += y.size(0)

    val_loss = total_loss / max(1, n_samps)
    val_oa   = correct / max(1, total)
    return val_loss, val_oa


# ---------------------------
# 滑窗整图推理（预测整幅图每个像素的类别 1..K；背景保持 0）
# ---------------------------
@torch.no_grad()
def infer_full_map(model, data1, data2, gt, ws=9, infer_batch=2048, device=torch.device('cpu')):
    H, W, C = data1.shape
    pad = ws // 2

    d1p = pad_with_reflection(data1, pad)
    d2p = pad_with_reflection(data2, pad)

    # 生成所有中心像素的行列
    coords = [(r, c) for r in range(H) for c in range(W)]
    preds = np.zeros(H * W, dtype=np.int16)

    def get_patch(r, c):
        rp, cp = r + pad, c + pad
        r0, r1 = rp - pad, rp + pad + 1
        c0, c1 = cp - pad, cp + pad + 1
        p1 = d1p[r0:r1, c0:c1, :]
        p2 = d2p[r0:r1, c0:c1, :]
        p1 = np.transpose(p1, (2, 0, 1)).astype(np.float32)
        p2 = np.transpose(p2, (2, 0, 1)).astype(np.float32)
        return p1, p2

    model.eval()
    i = 0
    while i < len(coords):
        batch_coords = coords[i:i+infer_batch]
        b = len(batch_coords)
        x1 = np.zeros((b, C, ws, ws), dtype=np.float32)
        x2 = np.zeros((b, C, ws, ws), dtype=np.float32)
        for j, (r, c) in enumerate(batch_coords):
            p1, p2 = get_patch(r, c)
            x1[j] = p1
            x2[j] = p2
        x1 = torch.from_numpy(x1).to(device)
        x2 = torch.from_numpy(x2).to(device)

        # x1 = torch.transpose(torch.flatten(x1, start_dim=2, end_dim=3), 1, 2) #GLAFORMER
        # x2 = torch.transpose(torch.flatten(x2, start_dim=2, end_dim=3), 1, 2)

        logits = model(x1, x2)  # (b, K)
        pred0 = torch.argmax(logits, dim=1).cpu().numpy()  # 0..K-1
        preds[i:i+b] = (pred0 + 1).astype(np.int16)       # 1..K
        i += b

    pred_map = preds.reshape(H, W)

    # 可选：用 gt 的背景覆盖（背景保持 0）
    pred_map = np.where(gt == 0, 0, pred_map)
    return pred_map


# ---------------------------
# 主流程
# ---------------------------
def main():
    # ====== 数据读取与预处理 ======
    data_path = os.path.join(r'C:\Users\12879\PycharmProjects\vmamba\venv\hehai\change')
    # 你也可以切换到其他数据集（注意标签格式一致：背景=0，前景=1..K）
    data1 = sio.loadmat(os.path.join(data_path, 'farm', 'farm06.mat'))['imgh']   # (H,W,C)
    data2 = sio.loadmat(os.path.join(data_path, 'farm', 'farm07.mat'))['imghl']  # (H,W,C)
    gt    = sio.loadmat(os.path.join(data_path, 'farm', 'label.mat'))['label']   # (H,W)
    #
    # data1 = sio.loadmat(os.path.join(data_path, 'Hermiston','hermiston2004.mat'))['HypeRvieW']
    # data2 = sio.loadmat(os.path.join(data_path, 'Hermiston','hermiston2007.mat'))['HypeRvieW']
    # gt = sio.loadmat(os.path.join(data_path, 'Hermiston','label.mat'))['gt5clasesHermiston']
    # gt[gt > 1] = 1

    # data1 = sio.loadmat(os.path.join(data_path, 'River','river_after.mat'))['river_after']
    # data2 = sio.loadmat(os.path.join(data_path, 'River','river_before.mat'))['river_before']
    # gt = sio.loadmat(os.path.join(data_path, 'River','groundtruth'))['lakelabel_v1']



    H, W, C = data1.shape
    print("Data shape:", data1.shape, data2.shape, gt.shape)

    # 标准化（分别对两期逐光谱标准化）
    d1_rs = data1.reshape(-1, C)
    d2_rs = data2.reshape(-1, C)
    sc1 = preprocessing.StandardScaler().fit(d1_rs)
    sc2 = preprocessing.StandardScaler().fit(d2_rs)
    data1_std = sc1.transform(d1_rs).reshape(H, W, C)
    data2_std = sc2.transform(d2_rs).reshape(H, W, C)

    # 统计类别
    class_count = 2
    gt_vec = gt.reshape(-1)
    for i in range(1, class_count+1):
        print(f"class {i} count = {(gt_vec == i).sum()}")

    # ====== 划分索引（按比例；背景不参与采样） ======
    samples_type = 'ratio'
    train_ratio = 0.01
    val_ratio   = 0.01

    train_idx_list = []
    for cls in range(1, class_count+1):
        idx_cls = np.where(gt_vec == cls)[0]
        n_cls   = len(idx_cls)
        k_cls   = int(np.ceil(n_cls * train_ratio))
        if k_cls > 0:
            sel = np.random.choice(n_cls, size=min(k_cls, n_cls), replace=False)
            train_idx_list.append(idx_cls[sel])
    train_index = np.concatenate(train_idx_list) if len(train_idx_list) else np.array([], dtype=np.int64)

    # 其余前景作为 val+test
    all_fg_index = np.where(gt_vec != 0)[0]
    rest_index   = np.setdiff1d(all_fg_index, train_index, assume_unique=False)
    val_count    = int(val_ratio * (len(rest_index) + len(train_index)))
    if val_count > 0 and len(rest_index) > 0:
        val_sel = np.random.choice(len(rest_index), size=min(val_count, len(rest_index)), replace=False)
        val_index  = rest_index[val_sel]
        test_index = np.setdiff1d(rest_index, val_index, assume_unique=False)
    else:
        val_index  = np.array([], dtype=np.int64)
        test_index = rest_index

    # ====== 构建 Patch 数据集与 DataLoader ======
    train_set = ChangePatchDataset(data1_std, data2_std, gt, train_index, ws=windowSize)
    val_set   = ChangePatchDataset(data1_std, data2_std, gt, val_index,   ws=windowSize)
    test_set  = ChangePatchDataset(data1_std, data2_std, gt, test_index,  ws=windowSize)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    # ====== 打印数据形状 ======
    print("Train size:", len(train_set), "Val size:", len(val_set), "Test size:", len(test_set))
    if len(train_set) > 0:
        ex_x1, ex_x2, ex_y = next(iter(train_loader))
        print("Single sample (C,H,W):", tuple(ex_x1.shape[1:]))
        print("Batch x1 shape:", tuple(ex_x1.shape),
              "Batch x2 shape:", tuple(ex_x2.shape),
              "Batch y shape:", tuple(ex_y.shape))

    # ====== 构建模型（自动适配前向方式） ======
    # 优先尝试 (bands=C, class_count=K, patch_size=ws) 的初始化，其次尝试 (H,W,C,K) 以兼容你的老构造
    net = modelpa.Net( C, class_count,H).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

    # ====== 训练 ======
    best_val_oa = -1.0
    best_val_loss = float('inf')
    best_state_by_oa = None
    best_state_by_loss = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        net.train()
        running_loss = 0.0

        for x1, x2, y in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = net(x1, x2)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)

        train_loss = running_loss / max(1, len(train_set))

        # ---- 验证：Loss + OA ----
        if epoch %10==0:
            val_loss, val_oa = evaluate_val(net, val_loader, device, criterion, ws=windowSize) if len(val_set) > 0 else (
            0.0, 0.0)
            print(f"Epoch [{epoch:03d}/{epochs}]  TrainLoss={train_loss:.6f}  ValLoss={val_loss:.6f}  ValOA={val_oa:.4f}")

            # ---- 根据 ValLoss 保存最优 ----
            if val_loss < best_val_loss:
                print('save')
                best_val_loss = val_loss
                best_state_by_loss = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                torch.save(best_state_by_loss, os.path.join(save_dir_models, "best_by_val_loss.pt"))


        scheduler.step()

    train_time = time.time() - t0
    print(
        f"Training finished. Best Val OA = {best_val_oa:.4f}, Best Val Loss = {best_val_loss:.6f}. Time: {train_time:.2f}s")

    # 训练完成后，按你需要的标准载回权重
    if best_state_by_loss is not None:
        net.load_state_dict({k: v.to(device) for k, v in best_state_by_loss.items()})
    torch.save(net.state_dict(), os.path.join(save_dir_models, "best_patch_model.pt"))  # 也保存一份当前内存中的

    # ====== 测试集 OA ======
    test_loss, test_oa = evaluate_val(net, test_loader, device, criterion, ws=windowSize)
    print(f"[Mini-batch Test]  Loss={test_loss:.6f}  OA={test_oa:.4f}")

    # ====== 整图滑窗推理 ======
    t1 = time.time()
    pred_map = infer_full_map(net, data1_std, data2_std, gt, ws=windowSize, infer_batch=2048, device=device)
    infer_time = time.time() - t1
    print(f"Full-image inference done in {infer_time:.2f}s")

    png_path = os.path.join(save_dir_results, f"{dataset_name}_pred.png")
    draw_classification_map(pred_map, png_path, scale=4.0, dpi=400)
    sio.savemat(os.path.join(save_dir_results, "pred.mat"), {"pred": pred_map.astype(np.int16)})
    print(f"Saved prediction map to: {png_path}")

    mask_fg = (gt != 0)
    y_true = gt[mask_fg].reshape(-1)
    y_pred = pred_map[mask_fg].reshape(-1)

    metrics_dict = compute_metrics_arrays(y_true, y_pred, class_count=class_count)

    # 打印与保存结果
    print("\n========== Full-image Metrics (foreground only) ==========")
    print(f"OA = {metrics_dict['OA']:.6f}")
    print(f"AA = {metrics_dict['AA']:.6f}")
    print(f"Kappa = {metrics_dict['Kappa']:.6f}")
    print("Per-class Acc:", metrics_dict["per_class_acc"])
    print("Per-class Precision:", metrics_dict["per_class_P"])
    print("Per-class Recall   :", metrics_dict["per_class_R"])
    print("Per-class F1       :", metrics_dict["per_class_F1"])
    print("Macro P/R/F1 = {:.6f} / {:.6f} / {:.6f}".format(
        metrics_dict["macro_P"], metrics_dict["macro_R"], metrics_dict["macro_F1"]
    ))
    print("=========================================================\n")

    # 写入 results 文本
    with open(os.path.join(save_dir_results, f"{dataset_name}_results.txt"), "a+", encoding="utf-8") as f:
        f.write('\n====================== PATCH TRAINING ======================\n')
        f.write(f"windowSize={windowSize}, epochs={epochs}, lr={lr}, gamma={gamma}, batch_size={batch_size}\n")
        f.write(f"Train size={len(train_set)}, Val size={len(val_set)}, Test size={len(test_set)}\n")
        f.write(f"Train time(s)={train_time:.2f}, Full-image infer time(s)={infer_time:.2f}\n")
        f.write(f"[Mini-batch Test] OA={test_oa:.6f}\n")
        f.write("---- Full-image metrics (foreground only) ----\n")
        f.write(f"OA={metrics_dict['OA']:.6f}\n")
        f.write(f"AA={metrics_dict['AA']:.6f}\n")
        f.write(f"Kappa={metrics_dict['Kappa']:.6f}\n")
        f.write(f"Per-class Acc={np.array2string(metrics_dict['per_class_acc'], precision=6)}\n")
        f.write(f"Per-class Precision={np.array2string(metrics_dict['per_class_P'], precision=6)}\n")
        f.write(f"Per-class Recall   ={np.array2string(metrics_dict['per_class_R'], precision=6)}\n")
        f.write(f"Per-class F1       ={np.array2string(metrics_dict['per_class_F1'], precision=6)}\n")
        f.write("Macro P/R/F1 = {:.6f} / {:.6f} / {:.6f}\n".format(
            metrics_dict["macro_P"], metrics_dict["macro_R"], metrics_dict["macro_F1"]
        ))

    print("All done. Results saved under:", save_dir_results)


if __name__ == "__main__":
    main()
