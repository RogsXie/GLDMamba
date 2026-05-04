import os
import time
import random
import warnings
warnings.filterwarnings("ignore")
from thop import profile
import numpy as np
import scipy.io as sio
from sklearn import preprocessing, metrics
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import GLDMambaP
import SSMIF
import  AIWSEN
import GLAFormer
import GTMSiam
import SSTFormer
import CSANet
# ---------------------------
# 设备
# ---------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------
# 种子列表 & 超参数
# ---------------------------
SEED_LIST   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 42]
windowSize  = 3
epochs      = 100
lr          = 1e-4
gamma       = 0.9
batch_size  = 128
train_ratio = 0.01
val_ratio   = 0.01

# ---------------------------
# 其他方法参数
# ---------------------------

# epochs      = 30
# windowSize = 5
# lr = 5e-3
# batch_size = 32
# epoch_number = 1

# epochs      = 100
# windowSize = 7
# lr = 5e-3
# batch_size = 64
# epoch_number = 1

# epochs      = 100
# windowSize = 5
# lr =   3e-5 #6e-4
# batch_size = 128
# # epoch_number = 1

# epochs      = 150
# windowSize = 5
# lr = 3e-4
# batch_size = 128
# epoch_number = 1

# epochs      = 50
# windowSize = 5
# lr = 5e-4
# batch_size = 64
# epoch_number = 1

# epochs      = 50
# windowSize = 7
# lr = 5e-4
# batch_size = 32
# epoch_number = 1




dataset_name     = "river"
save_dir_results = "results"
save_dir_models  = "model"
os.makedirs(save_dir_results, exist_ok=True)
os.makedirs(save_dir_models,  exist_ok=True)

data_path = r'C:\Users\12879\PycharmProjects\vmamba\venv\hehai\change'


# ===================================================================
# 工具函数
# ===================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_with_reflection(arr: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return arr
    return np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')


def linear_to_hw(idx: int, W: int):
    return int(idx // W), int(idx % W)


# ===================================================================
# 二分类变化检测指标（正类 = 变化 = label 2）
# ===================================================================
def compute_binary_cd_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               pos_label: int = 2):
    eps  = 1e-12
    pos  = (y_true == pos_label)
    neg  = ~pos
    pp   = (y_pred == pos_label)
    pn   = ~pp

    TP = int(np.sum(pos & pp))
    TN = int(np.sum(neg & pn))
    FP = int(np.sum(neg & pp))
    FN = int(np.sum(pos & pn))

    total = len(y_true)
    OA = (TP + TN) / total
    P  = TP / (TP + FP + eps)
    R  = TP / (TP + FN + eps)
    F1 = 2 * P * R / (P + R + eps)
    try:
        Kappa = metrics.cohen_kappa_score(y_true.astype(np.int16),
                                          y_pred.astype(np.int16))
    except Exception:
        Kappa = 0.0

    return dict(OA=float(OA), Kappa=float(Kappa),
                F1=float(F1), P=float(P), R=float(R),
                TP=TP, TN=TN, FP=FP, FN=FN)


# ===================================================================
# 绘制 TP/TN/FP/FN 四色图
# ===================================================================
def draw_cd_error_map(gt, pred_map, save_path, pos_label=2, dpi=400):
    H, W = gt.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    pos = (gt == pos_label)
    neg = (gt != 0) & (~pos)
    pred_pos = (pred_map == pos_label)
    pred_neg = ~pred_pos

    rgb[pos & pred_pos] = [255, 255, 255]  # TP → 白
    rgb[neg & pred_neg] = [0, 0, 0]  # TN → 黑
    rgb[neg & pred_pos] = [255, 0, 0]  # FP → 红
    rgb[pos & pred_neg] = [0, 0, 255]  # FN → 蓝
    rgb[gt == 0] = [128, 128, 128]  # 背景 → 灰

    # --- 核心修改部分 ---
    # 1. 设置 Figure 尺寸，确保比例与图像完全一致
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)

    # 2. 使用 add_axes 占满整个画布 [left, bottom, width, height]
    ax = fig.add_axes([0, 0, 1, 1])

    # 3. 关闭坐标轴及刻度
    ax.axis('off')

    # 4. 显示图像，设置 aspect='auto' 确保不拉伸，interpolation='none' 确保像素清晰
    ax.imshow(rgb, interpolation='none')

    # 5. 保存时强制 pad_inches 为 0
    fig.savefig(save_path, format='png', dpi=dpi, pad_inches=0)
    plt.close(fig)


# ===================================================================
# Patch 数据集
# ===================================================================
class ChangePatchDataset(Dataset):
    def __init__(self, data1, data2, gt, index_list, ws=9):
        self.H, self.W, self.C = data1.shape
        self.ws  = ws
        self.pad = ws // 2
        self.d1p = pad_with_reflection(data1, self.pad)
        self.d2p = pad_with_reflection(data2, self.pad)
        self.gtp = np.pad(gt, self.pad, mode='constant', constant_values=0)
        fg       = [int(i) for i in index_list
                    if gt[linear_to_hw(int(i), self.W)] != 0]
        self.idxs = np.array(fg, dtype=np.int64)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        idx    = int(self.idxs[i])
        r, c   = linear_to_hw(idx, self.W)
        rp, cp = r + self.pad, c + self.pad
        p1 = self.d1p[rp-self.pad:rp+self.pad+1, cp-self.pad:cp+self.pad+1, :]
        p2 = self.d2p[rp-self.pad:rp+self.pad+1, cp-self.pad:cp+self.pad+1, :]
        y  = int(self.gtp[rp, cp]) - 1
        p1 = np.transpose(p1, (2, 0, 1)).astype(np.float32)
        p2 = np.transpose(p2, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(p1), torch.from_numpy(p2), torch.tensor(y, dtype=torch.long)


# ===================================================================
# 验证
# ===================================================================
@torch.no_grad()
def evaluate_val(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x1, x2, y in loader:
        x1, x2, y  = x1.to(device), x2.to(device), y.to(device)
        logits      = model(x1, x2)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct    += (torch.argmax(logits, 1) == y).sum().item()
        total      += y.numel()
    return total_loss / max(1, total), correct / max(1, total)


# ===================================================================
# 滑窗整图推理
# ===================================================================
@torch.no_grad()
def infer_full_map(model, data1, data2, gt, ws=9, infer_batch=2048):
    H, W, C = data1.shape
    pad = ws // 2
    d1p = pad_with_reflection(data1, pad)
    d2p = pad_with_reflection(data2, pad)
    coords = [(r, c) for r in range(H) for c in range(W)]
    preds  = np.zeros(H * W, dtype=np.int16)
    model.eval()
    i = 0
    while i < len(coords):
        bc  = coords[i:i+infer_batch]
        b   = len(bc)
        bx1 = np.zeros((b, C, ws, ws), dtype=np.float32)
        bx2 = np.zeros((b, C, ws, ws), dtype=np.float32)
        for j, (r, c) in enumerate(bc):
            rp, cp = r + pad, c + pad
            bx1[j] = np.transpose(d1p[rp-pad:rp+pad+1, cp-pad:cp+pad+1, :], (2,0,1))
            bx2[j] = np.transpose(d2p[rp-pad:rp+pad+1, cp-pad:cp+pad+1, :], (2,0,1))
        logits = model(torch.from_numpy(bx1).to(device),
                       torch.from_numpy(bx2).to(device))
        preds[i:i+b] = (torch.argmax(logits, 1).cpu().numpy() + 1).astype(np.int16)
        i += b
    pred_map = preds.reshape(H, W)
    pred_map = np.where(gt == 0, 0, pred_map)
    return pred_map


# ===================================================================
# 单次实验
# ===================================================================
def run_one_seed(seed, run_idx, data1_std, data2_std, gt, H, W, C, class_count):
    set_seed(seed)
    print(f"\n[Run {run_idx+1:02d}/{len(SEED_LIST)}]  seed={seed}")

    gt_vec = gt.reshape(-1)

    # ---------- 划分索引 ----------
    train_idx_list = []
    for cls in range(1, class_count + 1):
        idx_cls = np.where(gt_vec == cls)[0]
        k       = max(1, int(np.ceil(len(idx_cls) * train_ratio)))
        sel     = np.random.choice(len(idx_cls), size=min(k, len(idx_cls)), replace=False)
        train_idx_list.append(idx_cls[sel])
    train_index  = np.concatenate(train_idx_list)
    all_fg_index = np.where(gt_vec != 0)[0]
    rest_index   = np.setdiff1d(all_fg_index, train_index)
    val_count    = int(val_ratio * len(all_fg_index))

    if val_count > 0 and len(rest_index) > 0:
        val_sel    = np.random.choice(len(rest_index),
                                      size=min(val_count, len(rest_index)), replace=False)
        val_index  = rest_index[val_sel]
        test_index = np.setdiff1d(rest_index, val_index)
    else:
        val_index  = np.array([], dtype=np.int64)
        test_index = rest_index

    # ---------- DataLoader ----------
    train_set = ChangePatchDataset(data1_std, data2_std, gt, train_index, ws=windowSize)
    val_set   = ChangePatchDataset(data1_std, data2_std, gt, val_index,   ws=windowSize)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)

    
    
    # ---------- 其他模型 ----------
    # net = SSMIF.SSMIF().to(device)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)

    # net = AIWSEN.AIWSEN(in_chans=C).to(device)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=5e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.9)

    # net = GLAFormer.GLAFormer(img_c=C,
    #             embed_dim=256,
    #             depth=6,
    #             num_heads=8,
    #             num_classes=2,
    # window_size=windowSize).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

    # net = GTMSiam.GTMSiam(1, C, 30, 30, 128).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    # net = SSTFormer.SSTViT(
    #         image_size = 5,
    #         near_band = 1,
    #         num_patches = C ,
    #         num_classes = 2,
    #         dim = 32,
    #         depth = 2,
    #         heads = 4,
    #         dim_head=16,
    #         mlp_dim = 8,
    #         b_dim = 512,
    #         b_depth = 3,
    #         b_heads = 8,
    #         b_dim_head= 32,
    #         b_mlp_head = 8,
    #         dropout = 0.2,
    #         emb_dropout = 0.1,
    #     ).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400 // 20, gamma=0.9)

    # criterion = nn.CrossEntropyLoss(reduce=False)

    # net = CSANet.Finalmodel().to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)


    criterion = nn.CrossEntropyLoss()
    net = modelpaatch.Net(C, class_count, H).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

    # ---------- 训练 ----------
    best_val_loss = float('inf')
    best_state    = None
    t_train_start = time.time()

    for epoch in range(1, epochs + 1):
        net.train()
        running_loss = 0.0
        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(x1, x2), y)

            # loss = torch.sum(loss)
            # loss.backward(retain_graph=True)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)

        if epoch % 10 == 0:
            val_loss, val_oa = (evaluate_val(net, val_loader, criterion)
                                if len(val_set) > 0 else (0., 0.))
            print(f"  Epoch [{epoch:03d}/{epochs}]  "
                  f"TrainLoss={running_loss/max(1,len(train_set)):.5f}  "
                  f"ValLoss={val_loss:.5f}  ValOA={val_oa:.4f}")
            if val_loss < best_val_loss:
                print("save")
                best_val_loss = val_loss
                best_state    = {k: v.detach().cpu() for k, v in net.state_dict().items()}

        scheduler.step()

    train_time = time.time() - t_train_start   # 训练耗时

    if best_state is not None:
        net.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # ---------- 整图推理 ----------
    t_test_start = time.time()
    pred_map = infer_full_map(net, data1_std, data2_std, gt,
                              ws=windowSize, infer_batch=2048)
    test_time = time.time() - t_test_start     # 推理耗时

    # ---------- 指标 ----------
    mask_fg = (gt != 0)
    m = compute_binary_cd_metrics(gt[mask_fg].reshape(-1),
                                  pred_map[mask_fg].reshape(-1), pos_label=2)

    # ---------- 四色图 ----------
    err_path = os.path.join(save_dir_results,
                            f"{dataset_name}_seed{seed}_run{run_idx+1:02d}_errormap.png")
    draw_cd_error_map(gt, pred_map, err_path, pos_label=2)

    # ---------- 本次结果打印 ----------
    print(f"\n  ── Run {run_idx+1:02d} / seed={seed} results ──")
    print(f"  OA    : {m['OA']*100:.2f}%")
    print(f"  Kappa : {m['Kappa']*100:.2f}%")
    print(f"  F1    : {m['F1']*100:.2f}%")
    print(f"  P     : {m['P']*100:.2f}%")
    print(f"  R     : {m['R']*100:.2f}%")
    print(f"  Train : {train_time:.2f}s  |  Test : {test_time:.4f}s")

    return m, train_time, test_time


# ===================================================================
# 主函数
# ===================================================================
def main():
    # -------- 读数据（只读一次） --------

    # data1 = sio.loadmat(os.path.join(data_path, 'farm', 'farm06.mat'))['imgh']
    # data2 = sio.loadmat(os.path.join(data_path, 'farm', 'farm07.mat'))['imghl']
    # gt = sio.loadmat(os.path.join(data_path, 'farm', 'label.mat'))['label']

    # data1 = sio.loadmat(os.path.join(data_path, 'Hermiston','hermiston2004.mat'))['HypeRvieW']
    # data2 = sio.loadmat(os.path.join(data_path, 'Hermiston','hermiston2007.mat'))['HypeRvieW']
    # gt = sio.loadmat(os.path.join(data_path, 'Hermiston','label.mat'))['gt5clasesHermiston']
    # gt[gt > 1] = 1

    data1 = sio.loadmat(os.path.join(data_path, 'River', 'river_after.mat'))['river_after']
    data2 = sio.loadmat(os.path.join(data_path, 'River', 'river_before.mat'))['river_before']
    gt    = sio.loadmat(os.path.join(data_path, 'River', 'groundtruth'))['lakelabel_v1']
    gt = gt + 1
    gt = np.where(gt == 1, 0, np.where(gt == 0, 1, gt))

    gt = gt + 1   # {1=不变, 2=变化}
    print("gt 唯一值:", np.unique(gt))

    H, W, C = data1.shape
    class_count = 2

    # -------- 标准化（只做一次） --------
    data1_std = preprocessing.StandardScaler() \
                    .fit_transform(data1.reshape(-1, C)).reshape(H, W, C)
    data2_std = preprocessing.StandardScaler() \
                    .fit_transform(data2.reshape(-1, C)).reshape(H, W, C)

    # -------- 收集结果的数组 --------
    OA_ALL         = np.zeros(len(SEED_LIST))
    KPP_ALL        = np.zeros(len(SEED_LIST))
    F1_ALL         = np.zeros(len(SEED_LIST))
    P_ALL          = np.zeros(len(SEED_LIST))
    R_ALL          = np.zeros(len(SEED_LIST))
    Train_Time_ALL = np.zeros(len(SEED_LIST))
    Test_Time_ALL  = np.zeros(len(SEED_LIST))

    curr_train_ratio = train_ratio

    # -------- 多种子循环 --------
    for run_idx, seed in enumerate(SEED_LIST):
        torch.cuda.empty_cache()
        m, train_time, test_time = run_one_seed(
            seed, run_idx, data1_std, data2_std, gt, H, W, C, class_count)

        OA_ALL[run_idx]         = m['OA']
        KPP_ALL[run_idx]        = m['Kappa']
        F1_ALL[run_idx]         = m['F1']
        P_ALL[run_idx]          = m['P']
        R_ALL[run_idx]          = m['R']
        Train_Time_ALL[run_idx] = train_time
        Test_Time_ALL[run_idx]  = test_time

    # -------- 输出 --------
    print("\n" + "═" * 60)
    print("train_ratio = {}".format(curr_train_ratio))
    print("{:.2f} ± {:.2f}".format(np.mean(OA_ALL  * 100), np.std(OA_ALL  * 100)))
    print("{:.2f} ± {:.2f}".format(np.mean(KPP_ALL * 100), np.std(KPP_ALL * 100)))
    print("{:.2f} ± {:.2f}".format(np.mean(F1_ALL  * 100), np.std(F1_ALL  * 100)))
    print("{:.2f} ± {:.2f}".format(np.mean(P_ALL   * 100), np.std(P_ALL   * 100)))
    print("{:.2f} ± {:.2f}".format(np.mean(R_ALL   * 100), np.std(R_ALL   * 100)))
    print("Average training time: {:.2f}s".format(np.mean(Train_Time_ALL)))
    print("Average testing  time: {:.4f}s".format(np.mean(Test_Time_ALL)))
    print("═" * 60)

    # -------- 写入文件（与 print 完全一致） --------
    result_txt = os.path.join(save_dir_results, f"{dataset_name}_multiseed_results.txt")
    with open(result_txt, 'a+', encoding='utf-8') as f:
        f.write("\n" + "═" * 60 + "\n")
        f.write("train_ratio = {}\n".format(curr_train_ratio))
        f.write("{:.2f} ± {:.2f}\n".format(np.mean(OA_ALL  * 100), np.std(OA_ALL  * 100)))
        f.write("{:.2f} ± {:.2f}\n".format(np.mean(KPP_ALL * 100), np.std(KPP_ALL * 100)))
        f.write("{:.2f} ± {:.2f}\n".format(np.mean(F1_ALL  * 100), np.std(F1_ALL  * 100)))
        f.write("{:.2f} ± {:.2f}\n".format(np.mean(P_ALL   * 100), np.std(P_ALL   * 100)))
        f.write("{:.2f} ± {:.2f}\n".format(np.mean(R_ALL   * 100), np.std(R_ALL   * 100)))
        f.write("Average training time: {:.2f}s\n".format(np.mean(Train_Time_ALL)))
        f.write("Average testing  time: {:.4f}s\n".format(np.mean(Test_Time_ALL)))
        f.write("═" * 60 + "\n")
    print(f"Results saved to: {result_txt}")


if __name__ == "__main__":
    main()
