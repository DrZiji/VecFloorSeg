import cv2
import os
import numpy as np
import networkx as nx

from PIL import Image
from tqdm import tqdm
from munkres import Munkres, print_matrix, make_cost_matrix


CUBI_OLDCLASSES = {
    'background': 0, 'Outdoor': 1, 'Wall': 2, 'Kitchen': 3, 'Dining': 4, 'Bedroom': 5,
    'Bath': 6, 'Entrance': 7, 'Railing': 8, 'Closet': 9, 'Garage': 10, 'Corridor': 11,
    'wall': 2
}


R2V_CLASSES = {
"wall": 0,
# "opening": [255, 60, 128], # opening有点麻烦，暂时不加了吧
"kitchen": 1,
"bathroom": 2,
"bedroom": 3,
"balcony": 4,
"closet": 5,
"hall": 6,
"background": 7,
"otherRoom": 8,
"Wall": 0,
"Railing": 0,
}
CUBI_OLDPALETTE = {
    # 旧版调色板
    (84, 162, 68): 0,
    (157, 27, 92): 1,
    (243, 168, 154): 2,
    (170, 230, 1): 3,
    (169, 154, 221): 4,
    (209, 122, 34): 5,
    (255, 203, 106): 6,
    (118, 202, 180): 7,
    (61, 150, 6): 8,
    (169, 235, 206): 9,
    (145, 181, 248): 10,
    (180, 216, 35): 11, # 35
    (138, 142, 3): 255
}

R2V_PALETTE = {
    (255, 255, 255): 0,
    # "opening": [255, 60, 128], # opening有点麻烦，暂时不加了吧
    (224, 255, 192): 1,
    (192, 255, 255): 2,
    (255, 224, 128): 3,
    (255, 224, 224): 4,
    (192, 192, 224): 5,
    (255, 160, 96): 6,
    (0, 0, 0): 7,
    (224, 224, 224): 8,
}

CLASSES = CUBI_OLDCLASSES
PALETTE = CUBI_OLDPALETTE


def RGB2PMode(img_array):
    output = np.zeros(img_array.shape[:2], dtype=np.uint8)
    for h in range(img_array.shape[0]):
        for w in range(img_array.shape[1]):
            color = tuple(img_array[h, w].tolist())
            output[h, w] = PALETTE[color]
    return output


def primitiveBipartite(gt, pred):
    # bipartite matching 版本，思路：首先获得gt和pred的每一个联通分量。

    TP_THRESHOLD = 0.5
    CLASS_IGNORE = 255
    # 思路：构建一个profit matrix,里面存储pred和gt的实例相互匹配时的奖励。
    # FP、FN分别是从TP_IoU_pred和gt中扔出来的结果
    TP_IoU_gt, TP_IoU_pred, FP, FN = {}, {}, {}, {}
    gt_arr, pred_arr = np.array(gt), np.array(pred)
    if len(pred_arr.shape) == 3 and pred_arr.shape[2] == 3:
        pred_arr = RGB2PMode(pred_arr)

    # 预处理，将墙和railing变成背景；
    for i in [CLASSES['Wall'], CLASSES['Railing'], CLASSES['wall']]:
        gt_arr[gt_arr == i] = CLASSES['background']
        pred_arr[pred_arr == i] = CLASSES['background']

    pred_arr[gt_arr == CLASS_IGNORE] = CLASSES['background']
    gt_arr[gt_arr == CLASS_IGNORE] = CLASSES['background']

    gt_arr = gt_arr.astype(np.uint8) # gt_arr是标签
    pred_arr = pred_arr.astype(np.uint8) # pred_arr是预测
    mask_gt = gt_arr != CLASSES['background']
    mask_pred = pred_arr != CLASSES['background']
    all_area_gt = np.sum(mask_gt.astype(int)) # 计算所有gt房间的面积
    all_area_pred = np.sum(mask_pred.astype(int)) # 计算pred房间的面积

    # 将background的标签置为0
    gt_arr[gt_arr == CLASSES['background']] = 0
    pred_arr[pred_arr == CLASSES['background']] = 0

    num_gts, num_preds = 0, 0
    gts, preds = np.zeros_like(gt_arr), np.zeros_like(pred_arr)
    # 计算实例的index矩阵
    for i in range(1, len(CLASSES)):
        class_gt = gt_arr == i
        class_pred = pred_arr == i
        if not (np.max(class_gt) or np.max(class_pred)):
            continue

        num_gt, gt_class, stats_gts, centroids_gts = cv2.connectedComponentsWithStats(class_gt.astype(np.uint8), connectivity=4)
        num_pred, pred_class, stats_preds, centroids_preds = \
            cv2.connectedComponentsWithStats(class_pred.astype(np.uint8), connectivity=4)

        gt_class[gt_class > 0] += num_gts
        pred_class[pred_class > 0] += num_preds

        num_gts = num_gts + num_gt - 1
        num_preds = num_preds + num_pred - 1

        gts = gts + gt_class
        preds = preds + pred_class

    # 思路：首先计算每一个gt到pred，以及pred到gt的连接，最后利用两个连接，获取最大图
    # 没有匹配上的pred就是FP，没有匹配上的gt是FN
    profit_matrix = np.zeros((num_gts + 1, num_preds + 1))
    area_insec_matrix = np.zeros_like(profit_matrix)

    gt2pred, pred2gt = {}, {}
    for i_gt in range(1, num_gts + 1):
        c_gt = gts == i_gt  # gt的房间（区域）实例，编号i_gt
        semantic_gt = gt_arr[c_gt] # 统计该房间的语义
        if semantic_gt.min() != semantic_gt.max():
            raise RuntimeError("one region has two semantics")
        semantic_gt = semantic_gt.max()
        area_c_gt = np.sum(c_gt.astype(float))
        region_pred = preds[c_gt]  # 该实例在pred中的区域
        values, counts = np.unique(region_pred, return_counts=True)

        # 若pred中对应的区域，包含有背景类联通分量，需要剔除
        mask = values != 0
        counts = counts[mask]
        values = values[mask]
        if values.shape[0] == 0: # 筛选之后shape为0，那么加入FN
            FN[i_gt] = (CLASSES['background'], 0, area_c_gt, 0)
            continue

        # 计算单向匹配距离，迅速想到的有两种办法:(1)和房间重合区域最大的;(2)和房间的IoU最大的
        # 实施(2)
        IoUs = []
        FN_flag = True
        for idx, j_pred in enumerate(values):
            c_pred = preds == int(j_pred)  # 获得pred中，编号为j_pred的连通区域
            semantic_pred = pred_arr[c_pred]
            if semantic_pred.min() != semantic_pred.max():
                raise RuntimeError("one pred region has two semantics")
            semantic_pred = semantic_pred.max()
            area_c_pred, area_insec = np.sum(c_pred.astype(float)), counts[idx]
            IoU = area_insec.astype(float) / \
              (area_c_gt + area_c_pred - area_insec)
            IoUs.append(IoU)
            if IoU >= TP_THRESHOLD and semantic_gt == semantic_pred:
                profit_matrix[i_gt, j_pred] = IoU
                area_insec_matrix[i_gt, j_pred] = area_insec
                if gt2pred.__contains__(i_gt):
                    gt2pred.append((j_pred + num_gts, IoU, area_c_gt, area_insec))
                else:
                    gt2pred = [(j_pred + num_gts, IoU, area_c_gt, area_insec)]

                FN_flag = False

        # 即使语义不匹配，FN也不会有增加
        IoUs = np.array(IoUs)
        max_IoU_idx, max_IoU = np.argmax(IoUs), np.max(IoUs)

        # if max_IoU < TP_THRESHOLD:
        if FN_flag:
            FN[i_gt] = \
                (values[max_IoU_idx], max_IoU, area_c_gt, counts[max_IoU_idx])

    # 计算TP_IoU of Pred，和gt的计算过程基本一致，但是要把标识反一下
    for i_pred in range(1, num_preds + 1):
        c_pred = preds == i_pred
        semantic_pred = pred_arr[c_pred]
        if semantic_pred.min() != semantic_pred.max():
            raise RuntimeError("one pred region has two semantics")
        semantic_pred = semantic_pred.max()
        area_c_pred = np.sum(c_pred.astype(float))
        region_gt = gts[c_pred]
        values, counts = np.unique(region_gt, return_counts=True)

        # 若区域中gt包含最多的是背景，需要剔除
        mask = values != CLASSES['background']
        counts = counts[mask]
        values = values[mask]
        if values.shape[0] == 0:
            FP[i_pred] = (CLASSES['background'], 0, area_c_pred, 0)
            continue

        IoUs = []

        FP_flag = True
        for idx, j_gt in enumerate(values):
            c_gt = gts == int(j_gt)
            semantic_gt = gt_arr[c_gt]
            if semantic_gt.min() != semantic_gt.max():
                raise RuntimeError("one region has two semantics")
            semantic_gt = semantic_gt.max()
            area_c_gt, area_insec = np.sum(c_gt.astype(float)), counts[idx]
            IoU = area_insec.astype(float) / \
              (area_c_gt + area_c_pred - area_insec)
            IoUs.append(IoU)
            if IoU >= TP_THRESHOLD and semantic_pred == semantic_gt:
                if profit_matrix[j_gt, i_pred] > 0:
                    assert profit_matrix[j_gt, i_pred] == IoU, 'IoU not consistent'
                else:
                    profit_matrix[j_gt, i_pred] = IoU
                    area_insec_matrix[j_gt, i_pred] = area_insec

                if pred2gt.__contains__(i_pred + num_gts):
                    pred2gt.append((j_gt, IoU, area_c_pred, area_insec))
                else:
                    pred2gt = [(j_gt, IoU, area_c_pred, area_insec)]

                FP_flag = False

        IoUs = np.array(IoUs)
        max_IoU_idx, max_IoU = np.argmax(IoUs), np.max(IoUs)

        # if max_IoU < TP_THRESHOLD:  # 同理，并没有增加FP的数量
        if FP_flag:
            FP[i_pred] = \
                (values[max_IoU_idx], max_IoU, area_c_pred, counts[max_IoU_idx])

    all_area_insec = area_insec_matrix.sum()

    # 利用munkres算法包
    profit_mat = profit_matrix[1:, 1:].tolist()
    if len(profit_mat) == 0: # 空集
        return 0,  all_area_gt, all_area_pred, all_area_insec

    cost_mat = make_cost_matrix(profit_mat, lambda cost: 1 - cost)
    m = Munkres()
    indexes = m.compute(cost_mat)
    valid_gts, valid_preds, valid_IoU = [], [], 0.
    for r, c in indexes:
        valid_gts.append(r + 1)
        valid_preds.append(c + 1)
        valid_IoU += profit_mat[r][c]

    TP = len(valid_gts)
    FP = num_preds - len(valid_preds)
    FN = num_gts - len(valid_gts)
    return valid_IoU / (TP + 0.5 * (FP + FN)), all_area_gt, all_area_pred, all_area_insec


if __name__ == "__main__":
    """
    对于基于图像的分割，如何统计pred中的实例是一个问题，有两种方式可以考虑；
    （1）直接基于pred，统计联通分量，每一个联通分量都是一个实例；
    （2）根据户型图的墙体、进行启发式划分之后，对划分后的每一个primitive进行voting;
    voting之后，在原图基础上提取联通分量，作为pred实例。
    """
    # pred_dir = 'E:\\ExpVisual-work2\\2022_1001_rf_2'
    # gt_dir = "E:\\Datasets\\CUBI_3\\ann_dir"
    # split = 'test'
    # gt_dir = os.path.join(gt_dir, split)
    # imgNames = os.listdir(gt_dir)

    # chamfers, normalized_chamfers, precisions, recalls, area_gts, area_preds = [], [], [], [], [], []
    # for imgName in tqdm(imgNames[:]):
    #     gt = Image.open(os.path.join(gt_dir, imgName))
    #     pred = Image.open(os.path.join(pred_dir, imgName))
    #     chamfer, normalized_chamfer, precision, recall, area_gt, area_pred = primitiveChamfer(gt, pred)
    #     chamfers.append(chamfer)
    #     normalized_chamfers.append(normalized_chamfer)
    #     precisions.append(precision)
    #     recalls.append(recall)
    #     area_gts.append(area_gt)
    #     area_preds.append(area_pred)
    #
    # # 计算所有gt的面积和，所有pred的面积和
    # all_area_gt = np.array(area_gts).sum()
    # all_area_pred = np.array(area_preds).sum()
    #
    # # chamfers最大值为1，表示了房间的整体性，没有经过房间面积的normalize
    # # precision,recall经过房间面积的normalize
    # chamfers = np.array(chamfers).mean() * 0.5
    # normalized_chamfers = np.array(normalized_chamfers).mean() * 0.5
    # normalized_ps, normalized_rs = [], []
    # for p, r, a_gt, a_pred in zip(precisions, recalls, area_gts, area_preds):
    #     normalized_p, normalized_r = p * a_pred / all_area_pred, r * a_gt / all_area_gt
    #     normalized_ps.append(normalized_p)
    #     normalized_rs.append(normalized_r)
    #
    # normalized_P, normalized_R = np.array(normalized_ps).sum(), \
    #                              np.array(normalized_rs).sum()
    # normalized_F1 = 2 * normalized_P * normalized_R / (normalized_P + normalized_R)
    #
    # chamfers = np.round(chamfers, 4)
    # normalized_P = np.round(normalized_P, 4)
    # normalized_R = np.round(normalized_R, 4)
    # normalized_F1 = np.round(normalized_F1, 4)
    #
    # print("metrics of room integrity:")
    # print("chamfer distance:{}, Normalized chamfer distance:{}, precision:{}, recall:{}, f1_score:{}".\
    #       format(chamfers, normalized_chamfers, normalized_P, normalized_R, normalized_F1))
    #
    # with open(os.path.join(pred_dir, 'metric_room_integrity.txt'), 'w') as f:
    #     f.write("chamfer distance:{}, Normalized chamfer distance:{}, precision:{}, recall:{}, f1_score:{}".\
    #       format(chamfers, normalized_chamfers, normalized_P, normalized_R, normalized_F1))


    pred_dir = 'E:\\ExpVisual-work2\\Rebuttal\\Uniform Sampling\\merge'
    gt_dir = "E:\\Datasets\\jp_CAD\\ann_dir_V3"
    # gt_dir = "E:\\Datasets\\CUBI_3\\ann_dir_1"
    # gt_dir = "E:\\Datasets\\CUBI_new_1\\ann_dir_1"
    split = 'test'
    gt_dir = os.path.join(gt_dir, split)
    imgNames = os.listdir(gt_dir)

    PQs, area_gts, area_preds, area_insecs = [], [], [], []
    for imgName in tqdm(imgNames[:]):
        gt = Image.open(os.path.join(gt_dir, imgName))
        pred = Image.open(os.path.join(pred_dir, imgName))
        PQ, area_gt, area_pred, area_insec = primitiveBipartite(gt, pred)
        PQs.append(PQ)
        area_gts.append(area_gt)
        area_preds.append(area_pred)
        area_insecs.append(area_insec)

    area_gts = np.array(area_gts)
    area_preds = np.array(area_preds)
    area_insecs = np.array(area_insecs)
    all_area_gts, all_area_preds, all_area_insec = area_gts.sum(), area_preds.sum(), area_insecs.sum()
    PQs = np.array(PQs)
    xNormalized_PQs = PQs.mean()
    normalized_PQs = np.sum(PQs * area_insecs) / all_area_insec
    print("metrics of room integrity(PQ):")
    print('Panoptic Integrity:{}'.format(normalized_PQs))

    with open(os.path.join(pred_dir, 'PQ_RI_{}.txt'.format(split)), 'w') as f:
        f.write("PQ distance: {}, normalized distance: {}".format(xNormalized_PQs, normalized_PQs))
