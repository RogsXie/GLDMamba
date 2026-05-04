import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
import modelmy429
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# ─── 设备 ──────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

Seed_List = [42]
# Seed_List = [24,27,28,29,42,43,42,27,27,42]
# Seed_List = [42,42,42,42,42,42,42,42,42,42]
# Seed_List = [9,8,7,6,5,4,3,2,1,0]
# Seed_List = [42,8,7,6,5,4,3,2,1,0]
# Seed_List=[40,41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
torch.cuda.empty_cache()


# ─── 灰度分类图（保留，用于兼容） ────────────────────────────────────────────
def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
    fig, ax = plt.subplots()
    numlabel = np.array(label, dtype=np.int16)
    ax.imshow(numlabel, cmap='gray')
    ax.set_axis_off()
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    plt.close(fig)


# ─── TP/TN/FP/FN 彩色分类图 ────────────────────────────────────────────────
def Draw_Map_TPTNFPFN(pred_2d, gt_full_2d, name: str,
                       changed_label: int = 2, scale: float = 4.0, dpi: int = 400):
    """
    TP  白 (255,255,255): gt=changed,   pred=changed
    TN  黑 (  0,  0,  0): gt=unchanged, pred=unchanged
    FP  红 (255,  0,  0): gt=unchanged, pred=changed
    FN  蓝 (  0,  0,255): gt=changed,   pred=unchanged
    BG  灰 (128,128,128): gt==0（背景，若有）
    """
    h, w     = gt_full_2d.shape
    pred_np  = np.array(pred_2d,    dtype=np.int16)
    gt_np    = np.array(gt_full_2d, dtype=np.int16)
    unchanged_label = 3 - changed_label   # 2类时: changed=2, unchanged=1

    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    TP_mask = (gt_np == changed_label)   & (pred_np == changed_label)
    TN_mask = (gt_np == unchanged_label) & (pred_np == unchanged_label)
    FP_mask = (gt_np != changed_label)   & (gt_np != 0) & (pred_np == changed_label)
    FN_mask = (gt_np == changed_label)   & (pred_np != changed_label)
    BG_mask = (gt_np == 0)

    rgb[TP_mask] = [255, 255, 255]   # 白
    rgb[TN_mask] = [  0,   0,   0]   # 黑
    rgb[FP_mask] = [255,   0,   0]   # 红
    rgb[FN_mask] = [  0,   0, 255]   # 蓝
    rgb[BG_mask] = [128, 128, 128]   # 灰

    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.set_axis_off()
    fig.set_size_inches(w * scale / dpi, h * scale / dpi)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    save_path = name + '_colored.png'
    fig.savefig(save_path, format='png', transparent=True, dpi=dpi, pad_inches=0)
    plt.close(fig)
    print("彩色 TP/TN/FP/FN 图已保存:", save_path)


# ─── One-Hot 编码 ─────────────────────────────────────────────────────────
def GT_To_One_Hot(gt, class_count):
    height, width = gt.shape
    GT_One_Hot = []
    for i in range(height):
        for j in range(width):
            temp = np.zeros(class_count, dtype=np.float32)
            lv = int(gt[i, j])
            if 1 <= lv <= class_count:
                temp[lv - 1] = 1
            GT_One_Hot.append(temp)
    return np.reshape(GT_One_Hot, [height, width, class_count])


# ─── 损失函数 ─────────────────────────────────────────────────────────────
def compute_loss(predict: torch.Tensor,
                 reallabel_onehot: torch.Tensor,
                 reallabel_mask: torch.Tensor):
    we = -torch.mul(reallabel_onehot, torch.log(predict + 1e-10))
    we = torch.mul(we, reallabel_mask)
    return torch.sum(we)
# ─── 改进的损失函数 (Hybrid Loss) ─────────────────────────────────────────────
# def compute_loss(predict, reallabel_onehot, reallabel_mask,
#                  alpha=None, gamma=2.0, dice_weight=0.5):
#     eps = 1e-10
#     # alpha: 每类的权重，形状 [class_count]，不传则自动从 batch 统计
#     mask_idx = (reallabel_mask[:, 0] == 1.0)
#     if mask_idx.sum() == 0:
#         return torch.tensor(0.0).to(predict.device)
#
#     valid_predict = predict[mask_idx]
#     valid_onehot  = reallabel_onehot[mask_idx]
#
#     # ── 自动计算 alpha（inverse frequency）──
#     class_counts = valid_onehot.sum(dim=0).clamp(min=1)           # [C]
#     inv_freq     = 1.0 / class_counts
#     alpha_w      = (inv_freq / inv_freq.sum()).to(predict.device)  # 归一化
#
#     # ── Focal Loss（逐类加权）──
#     pt        = torch.sum(valid_predict * valid_onehot, dim=1)
#     alpha_t   = torch.sum(alpha_w.unsqueeze(0) * valid_onehot, dim=1)
#     focal_loss = (-alpha_t * ((1 - pt) ** gamma) * torch.log(pt + eps)).mean()
#
#     # ── Dice Loss ──
#     intersection = torch.sum(valid_predict * valid_onehot, dim=0)
#     union        = torch.sum(valid_predict + valid_onehot, dim=0)
#     dice_loss    = 1 - ((2. * intersection + eps) / (union + eps)).mean()
#
#     return (1 - dice_weight) * focal_loss + dice_weight * dice_loss

# ─── 评估函数 ─────────────────────────────────────────────────────────────
def evaluate_performance(network_output, samples_gt, samples_gt_onehot,
                          require_AA_KPP=False, printFlag=True):
    """
    require_AA_KPP=False → 仅返回 OA（训练过程监控）
    require_AA_KPP=True  → 计算 OA/Kappa/F1/P/R，打印并写文件

    正类 = label 2（changed），负类 = label 1（unchanged），背景 = 0（不参与）
    F1/P/R 均针对"变化类"（changed，label=2）计算，与变化检测文献一致。
    """
    eps = 1e-12

    # ── 快速 OA（训练监控用） ──────────────────────────────────────────────
    if not require_AA_KPP:
        with torch.no_grad():
            avail   = (samples_gt != 0).float()
            correct = torch.where(
                torch.argmax(network_output, 1) == torch.argmax(samples_gt_onehot, 1),
                avail, torch.zeros_like(avail)
            ).sum()
            return correct.cpu() / (avail.sum() + eps)

    # ── 完整评估 ────────────────────────────────────────────────────────────
    with torch.no_grad():
        output_np = network_output.detach().cpu().numpy()
        gt_flat   = samples_gt.detach().cpu().numpy()          # 0 / 1 / 2

        output_np = np.reshape(output_np, [m * n, class_count])

        # 预测标签 (1-indexed: 1 or 2)
        pred_flat = (np.argmax(output_np, axis=-1) + 1).astype(np.int64)

        # 只对非背景（测试）像素计算
        mask    = gt_flat != 0
        gt_test = gt_flat[mask].astype(np.int64)    # 1 or 2
        pr_test = pred_flat[mask]                    # 1 or 2
        total   = int(mask.sum())

        # ── 混淆矩阵（正类 = 2, 负类 = 1） ──────────────────────────────
        TP = int(np.sum((gt_test == 2) & (pr_test == 2)))
        TN = int(np.sum((gt_test == 1) & (pr_test == 1)))
        FP = int(np.sum((gt_test == 1) & (pr_test == 2)))
        FN = int(np.sum((gt_test == 2) & (pr_test == 1)))

        OA    = (TP + TN) / (total + eps)
        P     = TP / (TP + FP + eps)
        R     = TP / (TP + FN + eps)
        F1    = 2 * P * R / (P + R + eps)
        kappa = float(metrics.cohen_kappa_score(gt_test.astype(np.int16),
                                                 pr_test.astype(np.int16)))

        if printFlag:
            print("─" * 60)
            print("  OA    = {:.6f}".format(OA))
            print("  Kappa = {:.6f}".format(kappa))
            print("  F1    = {:.6f}  (changed class)".format(F1))
            print("  P     = {:.6f}  (changed class)".format(P))
            print("  R     = {:.6f}  (changed class)".format(R))
            print("  TP={:d}  TN={:d}  FP={:d}  FN={:d}  total={:d}".format(
                TP, TN, FP, FN, total))
            print("─" * 60)

        # 追加全局列表
        OA_ALL.append(OA)
        KPP_ALL.append(kappa)
        F1_ALL.append(F1)
        P_ALL.append(P)
        R_ALL.append(R)

        # 写文件
        os.makedirs("results", exist_ok=True)
        with open(os.path.join('results', dataset_name + '_results.txt'),
                  'a+', encoding='utf-8') as f:
            f.write(
                '\n========================'
                ' lr={} epochs={} train_ratio={} val_ratio={}'
                ' ========================'.format(
                    learning_rate, max_epoch, train_ratio, val_ratio)
                + '\nOA    = {:.6f}'.format(OA)
                + '\nKappa = {:.6f}'.format(kappa)
                + '\nF1    = {:.6f}  (changed class)'.format(F1)
                + '\nP     = {:.6f}  (changed class)'.format(P)
                + '\nR     = {:.6f}  (changed class)'.format(R)
                + '\nTP={:d}  TN={:d}  FP={:d}  FN={:d}  total={:d}'.format(
                    TP, TN, FP, FN, total)
                + '\n'
            )

        return OA

def quick_f1(network_output, samples_gt):
    eps = 1e-12
    with torch.no_grad():
        output_np = network_output.detach().cpu().numpy().reshape(m * n, class_count)
        pred_flat = (np.argmax(output_np, axis=-1) + 1).astype(np.int64)
        gt_flat   = samples_gt.detach().cpu().numpy()
        mask      = gt_flat != 0
        gt_t, pr_t = gt_flat[mask].astype(np.int64), pred_flat[mask]
        TP = np.sum((gt_t == 2) & (pr_t == 2))
        FP = np.sum((gt_t == 1) & (pr_t == 2))
        FN = np.sum((gt_t == 2) & (pr_t == 1))
        P  = TP / (TP + FP + eps)
        R  = TP / (TP + FN + eps)
        return 2 * P * R / (P + R + eps)
# ══════════════════════════════════════════════════════════════════════════════
#  主循环
# ══════════════════════════════════════════════════════════════════════════════
for (FLAG, curr_train_ratio) in [(0, 100)]:
    torch.cuda.empty_cache()

    OA_ALL  = []
    KPP_ALL = []
    F1_ALL  = []
    P_ALL   = []
    R_ALL   = []
    Train_Time_ALL = []
    Test_Time_ALL  = []

    data_path = os.path.join(r'C:\Users\12879\PycharmProjects\vmamba\venv\hehai\change')

    data1 = sio.loadmat(os.path.join(data_path, 'farm', 'farm06.mat'))['imgh']
    data2 = sio.loadmat(os.path.join(data_path, 'farm', 'farm07.mat'))['imghl']
    gt = sio.loadmat(os.path.join(data_path, 'farm', 'label.mat'))['label']

    # data1 = sio.loadmat(os.path.join(data_path, 'Hermiston','hermiston2004.mat'))['HypeRvieW']
    # data2 = sio.loadmat(os.path.join(data_path, 'Hermiston','hermiston2007.mat'))['HypeRvieW']
    # gt = sio.loadmat(os.path.join(data_path, 'Hermiston','label.mat'))['gt5clasesHermiston']
    # gt[gt > 1] = 1

    # ── 加载数据集（以 River 为例） ────────────────────────────────────────
    # data1 = sio.loadmat(os.path.join(data_path, 'River', 'river_before.mat'))['river_before']
    # data2 = sio.loadmat(os.path.join(data_path, 'River', 'river_after.mat'))['river_after']
    # gt    = sio.loadmat(os.path.join(data_path, 'River', 'groundtruth'))['lakelabel_v1']
    # gt = gt + 1                                              # 0→1, 1→2  (现在1=changed_old,2=unchanged_old)
    # gt = np.where(gt == 1, 0, np.where(gt == 0, 1, gt))     # 交换，使 0 临时作为中间值


    # 最终再 +1 使标签从 1 开始，无 0
    gt = gt + 1
    # 此时唯一值应为 {1, 2}
    print("gt 的唯一值:", np.unique(gt))   # 期望: [1 2]

    # ── 保存 GT 可视化 ─────────────────────────────────────────────────────
    plt.figure(figsize=(gt.shape[1] * 4 / 400, gt.shape[0] * 4 / 400), dpi=400)
    plt.imshow(gt, cmap='gray')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('ground_truth.png', format='png', transparent=True, dpi=400)
    plt.close()

    # ── 参数 ───────────────────────────────────────────────────────────────
    samples_type = ['ratio', 'same_num'][FLAG]
    val_ratio    = 0.01
    class_count  = 2
    learning_rate = 0.0003
    max_epoch     = 200
    dataset_name  = "riv"
    train_ratio   = 0.01 if samples_type == "ratio" else curr_train_ratio
    train_samples_per_class = curr_train_ratio
    m, n, d = data1.shape

    # ── 归一化 ─────────────────────────────────────────────────────────────
    height, width, bands = data1.shape
    data1 = preprocessing.StandardScaler().fit_transform(
        data1.reshape(height * width, bands)).reshape(height, width, bands)
    data2 = preprocessing.StandardScaler().fit_transform(
        data2.reshape(height * width, bands)).reshape(height, width, bands)

    # ── 统计每类样本数 ─────────────────────────────────────────────────────
    gt_reshape = np.reshape(gt, [-1])
    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        print(f"Class {i+1}: {len(idx)} samples")

    # ══════════════════════════════════════════════════════════════════════
    for curr_seed in Seed_List:
        random.seed(curr_seed)
        gt_reshape = np.reshape(gt, [-1])   # 全图 GT，1 or 2（无 0）

        train_rand_idx = []
        if samples_type == 'ratio':
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                n_train = int(np.ceil(len(idx) * train_ratio))
                rand_idx = random.sample(range(len(idx)), n_train)
                train_rand_idx.append(idx[rand_idx])

            train_rand_idx  = np.array(train_rand_idx, dtype=object)
            train_data_index = np.concatenate([train_rand_idx[c]
                                               for c in range(train_rand_idx.shape[0])])
            sio.savemat('train_index.mat', {'index': train_data_index})

            train_data_index = set(train_data_index)
            all_data_index   = set(range(len(gt_reshape)))

            # River 无背景像素（gt 全为 1 or 2），background_idx 为空集
            background_idx   = set(np.where(gt_reshape == 0)[-1])
            test_data_index  = all_data_index - train_data_index - background_idx

            val_data_count   = int(val_ratio * (len(test_data_index) + len(train_data_index)))
            val_data_index   = set(random.sample(list(test_data_index), val_data_count))
            test_data_index  = test_data_index - val_data_index

            test_data_index  = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index   = list(val_data_index)
            sio.savemat('test_index.mat', {'index': test_data_index})

        # ── 构造 GT 掩码 ───────────────────────────────────────────────────
        def make_gt_mask(index_list):
            mask = np.zeros(gt_reshape.shape)
            for i in index_list:
                mask[i] = gt_reshape[i]
            return mask

        train_samples_gt = make_gt_mask(train_data_index).reshape(height, width)
        test_samples_gt  = make_gt_mask(test_data_index).reshape(height, width)
        val_samples_gt   = make_gt_mask(val_data_index).reshape(height, width)

        Test_GT = test_samples_gt.copy()   # 2D，供 Kappa 使用（已保留以备兼容）

        # ── One-Hot ────────────────────────────────────────────────────────
        def to_onehot_flat(gt_2d):
            oh = GT_To_One_Hot(gt_2d, class_count)
            return np.reshape(oh, [-1, class_count]).astype(np.float32)

        train_samples_gt_onehot = to_onehot_flat(train_samples_gt)
        test_samples_gt_onehot  = to_onehot_flat(test_samples_gt)
        val_samples_gt_onehot   = to_onehot_flat(val_samples_gt)

        # ── Label mask ─────────────────────────────────────────────────────
        def make_label_mask(gt_2d):
            gt_flat = gt_2d.reshape(-1)
            mask    = np.zeros([m * n, class_count], dtype=np.float32)
            for i in range(m * n):
                if gt_flat[i] != 0:
                    mask[i] = 1.0
            return mask

        train_label_mask = make_label_mask(train_samples_gt)
        test_label_mask  = make_label_mask(test_samples_gt)
        val_label_mask   = make_label_mask(val_samples_gt)

        # ── 转 Tensor ──────────────────────────────────────────────────────
        def to_tensor(arr):
            return torch.from_numpy(arr.astype(np.float32)).to(device)

        train_samples_gt        = to_tensor(train_samples_gt.reshape(-1))
        test_samples_gt         = to_tensor(test_samples_gt.reshape(-1))
        val_samples_gt          = to_tensor(val_samples_gt.reshape(-1))
        train_samples_gt_onehot = to_tensor(train_samples_gt_onehot)
        test_samples_gt_onehot  = to_tensor(test_samples_gt_onehot)
        val_samples_gt_onehot   = to_tensor(val_samples_gt_onehot)
        train_label_mask        = to_tensor(train_label_mask)
        test_label_mask         = to_tensor(test_label_mask)
        val_label_mask          = to_tensor(val_label_mask)

        net_input_1 = to_tensor(np.array(data1, np.float32))
        net_input_2 = to_tensor(np.array(data2, np.float32))

        # ── 构建模型 ───────────────────────────────────────────────────────
        net = modelmy429.Net(height, width, bands, class_count).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=50, T_mult=2, eta_min=1e-6
        # )

        best_loss = float('inf')
        net.train()
        tic1 = time.time()
        best_val_f1 = 0.0
        # ── 训练循环 ───────────────────────────────────────────────────────
        for i in range(max_epoch + 1):
            optimizer.zero_grad()
            output = net(net_input_1, net_input_2)
            loss   = compute_loss(output, train_samples_gt_onehot, train_label_mask)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % 1 == 0:
                with torch.no_grad():
                    net.eval()
                    val_f1 = quick_f1(output, val_samples_gt)
                    trainOA = evaluate_performance(output, train_samples_gt,
                                                   train_samples_gt_onehot)
                    valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
                    valOA   = evaluate_performance(output, val_samples_gt,
                                                   val_samples_gt_onehot)
                    print("Epoch {:4d}  train_loss={:.4f}  train_OA={:.4f}"
                          "  val_loss={:.4f}  val_OA={:.4f}".format(
                              i + 1, loss.item(), float(trainOA),
                              float(valloss), float(valOA)))

                    if valloss < best_loss:
                        best_loss = valloss
                        os.makedirs("model", exist_ok=True)
                        torch.save(net.state_dict(), "model/best_model.pt")
                        print("  → 保存最优模型 (val_loss={:.6f})".format(float(valloss)))

                    # if val_f1 > best_val_f1:  # ← 改这里
                    #     best_val_f1 = val_f1
                    #     os.makedirs("model", exist_ok=True)
                    #     torch.save(net.state_dict(), "model/best_model.pt")
                    #     print("  → 保存 best_model (val_F1={:.6f})".format(val_f1))

                torch.cuda.empty_cache()
                net.train()

        toc1 = time.time()
        training_time = toc1 - tic1
        Train_Time_ALL.append(training_time)
        print("\n\n==== 训练完成，开始测试 ====\n")

        # ── 测试 ───────────────────────────────────────────────────────────
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load("model/best_model.pt"))
            net.eval()

            tic2   = time.time()
            output = net(net_input_1, net_input_2)
            toc2   = time.time()

            testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
            testOA   = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot,
                                            require_AA_KPP=True, printFlag=True)
            print("test loss = {:.6f}".format(float(testloss)))

            testing_time = toc2 - tic2
            Test_Time_ALL.append(testing_time)

            # ── 预测图：原灰度图 + TP/TN/FP/FN 彩色图 ────────────────────
            classification_map = torch.argmax(output, 1) + 1   # 1 or 2
            if background_idx:
                bg_list = list(background_idx)
                classification_map[bg_list] = 0
            classification_map = classification_map.reshape([height, width]).cpu().numpy().astype(np.int16)

            # 全图 GT（用于彩色图）
            gt_full_2d = np.reshape(gt_reshape, [height, width])   # 1 or 2（River 无 0）

            save_stem = os.path.join("results", dataset_name + "_{:.4f}".format(float(testOA)))
            os.makedirs("results", exist_ok=True)

            # Draw_Classification_Map(classification_map, save_stem)
            Draw_Map_TPTNFPFN(classification_map, gt_full_2d, save_stem,
                               changed_label=2, scale=4.0, dpi=400)

            sio.savemat("results/pred.mat", {'pred': classification_map})

    # ══════════════════════════════════════════════════════════════════════
    #  汇总输出
    # ══════════════════════════════════════════════════════════════════════
    torch.cuda.empty_cache()
    del net

    OA_ALL  = np.array(OA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    F1_ALL  = np.array(F1_ALL)
    P_ALL   = np.array(P_ALL)
    R_ALL   = np.array(R_ALL)

    print("\n" + "═" * 60)
    print("train_ratio = {}".format(curr_train_ratio))
    print("{:.2f} ± {:.2f}".format(np.mean(OA_ALL*100),  np.std(OA_ALL*100)))
    print("{:.2f} ± {:.2f}".format(np.mean(KPP_ALL*100), np.std(KPP_ALL*100)))
    print("{:.2f} ± {:.2f}".format(np.mean(F1_ALL*100),  np.std(F1_ALL*100)))
    print("{:.2f} ± {:.2f}".format(np.mean(P_ALL*100),   np.std(P_ALL*100)))
    print("{:.2f} ± {:.2f}".format(np.mean(R_ALL*100),   np.std(R_ALL*100)))
    print("Average training time: {:.2f}s".format(np.mean(Train_Time_ALL)))
    print("Average testing  time: {:.4f}s".format(np.mean(Test_Time_ALL)))
    print("═" * 60)

    with open(os.path.join('results', dataset_name + '_results.txt'),
              'a+', encoding='utf-8') as f:
        f.write(
            '\n\n════════════════════════════════════════'
            '\ntrain_ratio = {}'.format(curr_train_ratio)
            + '\nOA    = {:.6f} ± {:.6f}'.format(np.mean(OA_ALL),  np.std(OA_ALL))
            + '\nKappa = {:.6f} ± {:.6f}'.format(np.mean(KPP_ALL), np.std(KPP_ALL))
            + '\nF1    = {:.6f} ± {:.6f}'.format(np.mean(F1_ALL),  np.std(F1_ALL))
            + '\nP     = {:.6f} ± {:.6f}'.format(np.mean(P_ALL),   np.std(P_ALL))
            + '\nR     = {:.6f} ± {:.6f}'.format(np.mean(R_ALL),   np.std(R_ALL))
            + '\nAverage training time: {:.2f}s'.format(np.mean(Train_Time_ALL))
            + '\nAverage testing  time: {:.4f}s'.format(np.mean(Test_Time_ALL))
            + '\n'
        )
