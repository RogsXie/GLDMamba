import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import spectral as spy
from scipy.io import loadmat
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
import modelmy
import torch.nn as nn
import matplotlib
import torch
import cv2
import numpy as np

matplotlib.use('Agg')  # 非交互式后端，避免PyCharm兼容性问题
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

Seed_List = [42]
torch.cuda.empty_cache()
# 画图
def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    numlabel = numlabel.astype(np.int16)
    plt.imshow(numlabel, cmap='gray')
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    pass

def GT_To_One_Hot(gt, class_count):
    height, width = gt.shape
    GT_One_Hot = []

    for i in range(height):
        for j in range(width):
            temp = np.zeros(class_count, dtype=np.float32)
            label_val = int(gt[i, j])

            if label_val == 1:
                temp[0] = 1
            elif label_val == 2:
                temp[1] = 1
            GT_One_Hot.append(temp)

    return np.reshape(GT_One_Hot, [height, width, class_count])

def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
    real_labels = reallabel_onehot
    we = -torch.mul(real_labels, torch.log(predict + 1e-10))
    we = torch.mul(we, reallabel_mask)
    pool_cross_entropy = torch.sum(we)
    return pool_cross_entropy


def compute_crossentropy(index, data, gt, criteon):
    data_new = torch.cat([data[ind, :].unsqueeze(0) for ind in index], dim=0)
    return criteon(data_new, gt.long())

def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, require_AA_KPP=False,
                         printFlag=True):
    eps = 1e-12

    if False == require_AA_KPP:
        with torch.no_grad():
            available_label_idx = (train_samples_gt != 0).float()
            available_label_count = available_label_idx.sum()
            # 修复：zeros 未定义 -> 用 zeros_like
            correct_prediction = torch.where(
                torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                available_label_idx, torch.zeros_like(available_label_idx)
            ).sum()
            OA = correct_prediction.cpu() / (available_label_count + eps)
            return OA
    else:
        with torch.no_grad():
            # ============ OA（只统计非背景样本） ============
            available_label_idx = (train_samples_gt != 0).float()
            available_label_count = available_label_idx.sum()
            correct_prediction = torch.where(
                torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                available_label_idx, torch.zeros_like(available_label_idx)
            ).sum()
            OA = (correct_prediction.cpu() / (available_label_count + eps)).cpu().numpy()

            # 转 numpy
            zero_vector = np.zeros([class_count], dtype=np.float32)
            output_data = network_output.detach().cpu().numpy()
            train_samples_gt = train_samples_gt.detach().cpu().numpy()
            train_samples_gt_onehot = train_samples_gt_onehot.detach().cpu().numpy()

            # ============ 取预测标签 idx：0=背景, 1..class_count=前景各类 ============
            output_data = np.reshape(output_data, [m * n, class_count])
            idx = np.argmax(output_data, axis=-1)           # 0..class_count-1
            # 非全零行 -> 将类别索引 +1（映射到 1..class_count），全零保持 0（背景）
            non_zero_rows = ~np.all(output_data == zero_vector, axis=1)
            idx = idx.astype(np.int64)
            idx[non_zero_rows] += 1

            # ============ 每类准确率（不含背景） ============
            count_perclass = np.zeros([class_count], dtype=np.int64)
            correct_perclass = np.zeros([class_count], dtype=np.int64)
            for x in range(len(train_samples_gt)):
                gt = int(train_samples_gt[x])  # 0=背景, 1..class_count
                if gt != 0 and 1 <= gt <= class_count:
                    count_perclass[gt - 1] += 1
                    if idx[x] == gt:
                        correct_perclass[gt - 1] += 1

            # 防止除零；AA 只对有样本的类求平均
            test_AC_list = correct_perclass / (count_perclass + eps)
            valid_cls_mask = count_perclass > 0
            test_AA = np.mean(test_AC_list[valid_cls_mask]) if np.any(valid_cls_mask) else 0.0

            # ============ F1 计算 ============

            # 1) 逐类（不含背景）：对每类计算 P/R/F1，再做宏平均
            TP_c = np.zeros(class_count, dtype=np.int64)
            FP_c = np.zeros(class_count, dtype=np.int64)
            FN_c = np.zeros(class_count, dtype=np.int64)
            support_c = np.zeros(class_count, dtype=np.int64)

            for x in range(len(train_samples_gt)):
                gt = int(train_samples_gt[x])   # 0=背景, 1..class_count
                pr = int(idx[x])                # 0=背景, 1..class_count

                if gt != 0:
                    support_c[gt - 1] += 1
                    if pr == gt:
                        TP_c[gt - 1] += 1
                    else:
                        FN_c[gt - 1] += 1
                        if pr != 0:
                            # 预测成了其他前景类 -> 该预测类的 FP +1
                            FP_c[pr - 1] += 1
                else:
                    # 真实背景；若预测为某前景类 pr>0，则该类 FP +1
                    if pr != 0:
                        FP_c[pr - 1] += 1

            P_c = TP_c / (TP_c + FP_c + eps)
            R_c = TP_c / (TP_c + FN_c + eps)
            F1_c = 2 * P_c * R_c / (P_c + R_c + eps)

            valid = support_c > 0
            macro_P = np.mean(P_c[valid]) if np.any(valid) else 0.0
            macro_R = np.mean(R_c[valid]) if np.any(valid) else 0.0
            macro_F1 = np.mean(F1_c[valid]) if np.any(valid) else 0.0

            # 2) 二分类（前景 vs 背景）：把所有非背景并为正类
            is_fg_gt = (train_samples_gt != 0)
            is_fg_pred = (idx != 0)

            TP_bin = np.sum(is_fg_gt & is_fg_pred)
            FP_bin = np.sum(~is_fg_gt & is_fg_pred)
            FN_bin = np.sum(is_fg_gt & ~is_fg_pred)

            P_bin = TP_bin / (TP_bin + FP_bin + eps)
            R_bin = TP_bin / (TP_bin + FN_bin + eps)
            F1_bin = 2 * P_bin * R_bin / (P_bin + R_bin + eps)

            # ============ Kappa（对非背景像素） ============
            # 注意：这里重新用 0..class_count-1 的 argmax 索引，
            # 再 +1 以对齐真实标签（1..class_count），并只统计 Test_GT != 0 的位置
            output_argmax = np.argmax(np.reshape(output_data, [m * n, class_count]), axis=-1)
            output_argmax = np.reshape(output_argmax, [m, n])
            test_pre_label_list = []
            test_real_label_list = []
            for ii in range(m):
                for jj in range(n):
                    if Test_GT[ii][jj] != 0:
                        test_pre_label_list.append(output_argmax[ii][jj] + 1)
                        test_real_label_list.append(Test_GT[ii][jj])
            test_pre_label_list = np.array(test_pre_label_list)
            test_real_label_list = np.array(test_real_label_list)
            test_kpp = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                 test_real_label_list.astype(np.int16))

            # ============ 打印 ============
            if printFlag:
                print("test OA={:.6f}, AA={:.6f}, kpp={:.6f}".format(float(OA), float(test_AA), float(test_kpp)))
                print('acc per class:', test_AC_list)
                print('Per-class Precision:', P_c)
                print('Per-class Recall   :', R_c)
                print('Per-class F1       :', F1_c)
                print('Macro  P/R/F1 = {:.6f} / {:.6f} / {:.6f}'.format(macro_P, macro_R, macro_F1))
                print('Binary FG-vs-BG  P/R/F1 = {:.6f} / {:.6f} / {:.6f}'.format(P_bin, R_bin, F1_bin))

            # 结果收集（保持你原结构）
            OA_ALL.append(OA)
            AA_ALL.append(test_AA)
            KPP_ALL.append(test_kpp)
            AVG_ALL.append(test_AC_list)

            # ============ 存盘 ============
            os.makedirs("results", exist_ok=True)
            with open(os.path.join('results', dataset_name + '_results.txt'), 'a+', encoding='utf-8') as f:
                str_results = '\n======================' \
                              + " learning rate=" + str(learning_rate) \
                              + " epochs=" + str(max_epoch) \
                              + " train ratio=" + str(train_ratio) \
                              + " val ratio=" + str(val_ratio) \
                              + " ======================" \
                              + "\nOA=" + str(float(OA)) \
                              + "\nAA=" + str(float(test_AA)) \
                              + '\nkpp=' + str(float(test_kpp)) \
                              + '\nacc per class: ' + np.array2string(test_AC_list, precision=6) \
                              + '\nper-class Precision: ' + np.array2string(P_c, precision=6) \
                              + '\nper-class Recall   : ' + np.array2string(R_c, precision=6) \
                              + '\nper-class F1       : ' + np.array2string(F1_c, precision=6) \
                              + '\nMacro P/R/F1: {:.6f}/{:.6f}/{:.6f}'.format(macro_P, macro_R, macro_F1) \
                              + '\nBinary FG-vs-BG P/R/F1: {:.6f}/{:.6f}/{:.6f}'.format(P_bin, R_bin, F1_bin) \
                              + "\n"
                f.write(str_results)

            return OA

def save_train_gt_image_400dpi(train_samples_gt, gt_reshape, train_data_index, height, width, save_path):

    vis_img = np.zeros((height, width, 3), dtype=np.uint8)

    train_mask_flat = train_samples_gt.reshape(-1)
    train_set = set(train_data_index)

    for idx in range(len(gt_reshape)):
        h = idx // width
        w = idx % width

        if idx not in train_set:
            vis_img[h, w] = (34, 51, 226)
        else:
            if train_mask_flat[idx] == 1:
                vis_img[h, w] = (255, 255, 255)  # White
            else:
                vis_img[h, w] = (0, 0, 0)        # Black

    # 使用 matplotlib 保存为 400 DPI
    plt.figure(figsize=(width/100, height/100), dpi=100)
    plt.imshow(vis_img)
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()

    print("训练集 GT 可视化图已以 400 DPI 保存到：", save_path)



for (FLAG, curr_train_ratio) in [(0, 100)]:
    torch.cuda.empty_cache()
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []

    data_path = os.path.join(r'C:\Users\12879\PycharmProjects\vmamba\venv\hehai\change')

    # data1 = sio.loadmat(os.path.join(data_path, 'farm', 'farm06.mat'))['imgh']
    # data2 = sio.loadmat(os.path.join(data_path, 'farm', 'farm07.mat'))['imghl']
    # gt = sio.loadmat(os.path.join(data_path, 'farm', 'label.mat'))['label']

    data1 = sio.loadmat(os.path.join(data_path, 'Hermiston','hermiston2004.mat'))['HypeRvieW']
    data2 = sio.loadmat(os.path.join(data_path, 'Hermiston','hermiston2007.mat'))['HypeRvieW']
    gt = sio.loadmat(os.path.join(data_path, 'Hermiston','label.mat'))['gt5clasesHermiston']
    gt[gt > 1] = 1

    # data1 = sio.loadmat(os.path.join(data_path, 'china', 'China1.mat'))['T1']
    # data2 = sio.loadmat(os.path.join(data_path, 'china', 'China2.mat'))['T2']
    # gt = sio.loadmat(os.path.join(data_path, 'china', 'GT.mat'))['GT']

    # data1 = sio.loadmat(os.path.join(data_path, 'River', 'river_before.mat'))['river_before']
    # data2 = sio.loadmat(os.path.join(data_path, 'River','river_after.mat'))['river_after']
    # gt = sio.loadmat(os.path.join(data_path, 'River','groundtruth'))['lakelabel_v1']


    # data1 = sio.loadmat(os.path.join(data_path, 'bayArea', 'Bay_Area_2013.mat'))['HypeRvieW']
    # data2 = sio.loadmat(os.path.join(data_path, 'bayArea','Bay_Area_2015.mat'))['HypeRvieW']
    # gt = sio.loadmat(os.path.join(data_path, 'bayArea', 'bayArea_gtChanges2.mat'))['HypeRvieW']

    # data1 = sio.loadmat(os.path.join(data_path, 'farm420/farm420(1).mat'))['img_1']
    # data2 = sio.loadmat(os.path.join(data_path, 'farm420/farm420(1).mat'))['img_2']
    # gt = sio.loadmat(os.path.join(data_path, 'farm420/farm420(1).mat'))['GT']

    # """
    plt.figure(figsize=(gt.shape[1] * 4 / 400, gt.shape[0] * 4 / 400), dpi=400)
    plt.imshow(gt, cmap='gray')  # 显示灰度图像
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('ground_truth6.png', format='png', transparent=True, dpi=400)
    # """

    gt = gt + 1
    print("gt的唯一值:", np.unique(gt))

    samples_type = ['ratio', 'same_num'][FLAG]  # ratio or number

    # parameter preset
    val_ratio = 0.01
    class_count = 2  # class
    learning_rate = 0.0001  # learning rate
    max_epoch = 300  # iterations
    dataset_name = "riv"  # dataset name
    train_ratio = 0.01 if samples_type == "ratio" else curr_train_ratio #river 3.6% fram 0.05% hermiston 1%
    train_samples_per_class = curr_train_ratio
    val_samples = class_count
    m, n, d = data1.shape  # shape of dataset

    # standardization 归一化
    height, width, bands = data1.shape
    data1 = np.reshape(data1, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data1 = minMax.fit_transform(data1)
    data1 = np.reshape(data1, [height, width, bands])
    # orig_data = data2
    height, width, bands = data2.shape
    data2 = np.reshape(data2, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data2 = minMax.fit_transform(data2)
    data2 = np.reshape(data2, [height, width, bands])

    # print the number of samples per class
    gt_reshape = np.reshape(gt, [-1])
    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        print(samplesCount)

    for curr_seed in Seed_List:
        random.seed(curr_seed)
        gt_reshape = np.reshape(gt, [-1])
        train_rand_idx = []
        val_rand_idx = []
        if samples_type == 'ratio':  # Take a certain percentage of training
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                rand_list = [i for i in range(samplesCount)]
                rand_idx = random.sample(rand_list,
                                         np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)

                trainsamplesCount = len(rand_idx)
                print(trainsamplesCount)
                rand_real_idx_per_class = idx[rand_idx]
                train_rand_idx.append(rand_real_idx_per_class)
            # train_rand_idx = np.array(train_rand_idx)
            train_rand_idx = np.array(train_rand_idx, dtype=object)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)
            sio.savemat('train_index.mat', {'index': train_data_index})

            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)

            # the index of the background
            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index - background_idx

            # the validation set
            val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)
            test_data_index = test_data_index - val_data_index

            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)
            sio.savemat('test_index.mat', {'index': test_data_index})


        # train set 训练集标签
        train_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(train_data_index)):
            train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
            pass

        # test set 测试集标签
        test_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(test_data_index)):
            test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
            pass

        Test_GT = np.reshape(test_samples_gt, [m, n])  # 测试样本图

        # validation set 验证集标签
        val_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(val_data_index)):
            val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
            pass

        train_samples_gt = np.reshape(train_samples_gt, [height, width])
        test_samples_gt = np.reshape(test_samples_gt, [height, width])
        val_samples_gt = np.reshape(val_samples_gt, [height, width])

        train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
        test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)
        val_samples_gt_onehot = GT_To_One_Hot(val_samples_gt, class_count)

        train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
        test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)
        val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, class_count]).astype(int)

        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m * n, class_count])

        # test set
        test_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m * n, class_count])

        # validation set
        val_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        for i in range(m * n):
            if val_samples_gt[i] != 0:
                val_label_mask[i] = temp_ones
        val_label_mask = np.reshape(val_label_mask, [m * n, class_count])

        train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
        test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
        val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)

        train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
        test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
        val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)

        train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
        val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

        net_input_1 = np.array(data1, np.float32)
        net_input_1 = torch.from_numpy(net_input_1.astype(np.float32)).to(device)
        net_input_2 = np.array(data2, np.float32)
        net_input_2 = torch.from_numpy(net_input_2.astype(np.float32)).to(device)

        zeros = torch.zeros([m * n]).to(device).float()
        net = modelmy.Net(height, width, bands, class_count)
        net.to(device)


        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
        best_loss = 99999
        best_oa=0
        net.train()
        tic1 = time.time()
        gt_tensor = torch.from_numpy(gt_reshape).to(device) - torch.ones_like(torch.from_numpy(gt_reshape)).to(device)
        gt_new = torch.cat([gt_tensor[ind].unsqueeze(0) for ind in train_data_index], dim=0)
        for i in range(max_epoch + 1):
            optimizer.zero_grad()  # zero the gradient buffers
            output = net(net_input_1, net_input_2)
            loss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
            loss.backward(retain_graph=False)
            optimizer.step()
            scheduler.step()
            if i % 10 == 0:
                with torch.no_grad():
                    net.eval()
                    trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
                    trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
                    valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
                    valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
                    print(
                        "{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA,
                                                                                         valloss, valOA))

                    if valloss < best_loss:
                            best_loss = valloss
                            os.makedirs("model", exist_ok=True)
                            torch.save(net.state_dict(), "model/best_model.pt")
                            print('save model...')

                torch.cuda.empty_cache()
                net.train()
        toc1 = time.time()
        print("\n\n====================training done. starting evaluation...========================\n")
        training_time = toc1 - tic1
        Train_Time_ALL.append(training_time)

        # testing
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load("model/best_model.pt"))
            net.eval()
            tic2 = time.time()
            output = net(net_input_1, net_input_2)
            # output = net(net_input_1, net_input_2)
            toc2 = time.time()
            testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
            testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot, require_AA_KPP=True,
                                          printFlag=False)
            print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))

            testing_time = toc2 - tic2
            Test_Time_ALL.append(testing_time)

            classification_map = torch.argmax(output, 1) + 1
            background_idx = list(background_idx)
            classification_map[background_idx] = 0
            classification_map = classification_map.reshape([height, width]).cpu()
            Draw_Classification_Map(classification_map, "results/" + dataset_name + str(testOA))
            pred = np.array(classification_map)
            sio.savemat("results/pred.mat", {'pred': pred})

    torch.cuda.empty_cache()
    del net

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)
    print("\ntrain_ratio={}".format(curr_train_ratio),
          "\n==============================================================================")
    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))

    f = open('results/' + dataset_name + '_results.txt', 'a+')
    str_results = '\n\n************************************************' \
                  + "\ntrain_ratio={}".format(curr_train_ratio) \
                  + '\nOA=' + str(np.mean(OA_ALL)) + '+-' + str(np.std(OA_ALL)) \
                  + '\nAA=' + str(np.mean(AA_ALL)) + '+-' + str(np.std(AA_ALL)) \
                  + '\nKpp=' + str(np.mean(KPP_ALL)) + '+-' + str(np.std(KPP_ALL)) \
                  + '\nAVG=' + str(np.mean(AVG_ALL, 0)) + '+-' + str(np.std(AVG_ALL, 0)) \
                  + "\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
                  + "\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
    f.write(str_results)
    f.close()