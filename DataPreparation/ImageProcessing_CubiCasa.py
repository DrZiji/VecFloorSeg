import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from DataPreparation.SvgProcessing_CubiCasa import SVGParserCUBI
from Utils.graphicsUtilsRe import isLineIntersection
from Utils.extendWall import extendCornerWall
from Utils.triPlot import plot as myPlot, segments as mySegments

reSize = 512

REFER_NAME = 'wall_svg_1.png'
TARGET_DIR = 'CUBI_new_1'


def alterLabel(imgArray, lblArray):
    # 目标：找到由imgArray中的线段划分出来的联通区域，在联通区域中做vote操作
    # 从而将联通区域的label设置为同一种。
    """
    :param imgArray: RGBA格式的 ndarray
    :param lblArray: 单通道的ndarray
    :return:
    """
    # 第一步，imgArray二值化，非边界区域是纯白色
    lblFinal = np.zeros_like(lblArray)
    B, G, R = imgArray[:, :, 0], imgArray[:, :, 1], imgArray[:, :, 2]
    BMask, GMask, RMask = B == 255, G == 255, R == 255
    mask = np.logical_and(BMask, GMask)  # 0表示了被划分出去的边界
    mask = np.logical_and(mask, RMask).astype(np.uint8)
    # 利用mask，寻找联通区域
    lblBdry = np.copy(lblArray)
    lblFinal = lblFinal + lblBdry * (1 - mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    for i in range(1, num_labels):
        component = labels == i
        lbl_component = lblArray[component]
        lbls, lblCounts = np.unique(lbl_component, return_counts=True)
        tempIdx = np.argmax(lblCounts)
        temp = lbls[tempIdx]
        lblFinal[component] = temp

    return lblFinal.astype(np.uint8)


def drawPrimitives(ax, primitives, vertices, color='r'):
    edges = []
    for p in primitives:
        edges.extend(
            [(p[0], p[-1]), (p[1], p[0]), (p[2], p[1]), (p[-1], p[2])]
        )
    draw_dict = dict(vertices=vertices, segments=edges)
    mySegments(ax, color=color, **draw_dict)


def worker(filePth, imgSaveDir, labelSaveDir):
    imgName = filePth.split('\\')[-2]
    s = 50
    p = SVGParserCUBI(filePth + 'model.svg')
    verticesDict, pDoors, pWindows = p.getWallShape()

    lblImg = Image.open(filePth + REFER_NAME)
    width, height = lblImg.width, lblImg.height
    w, h = width * 1., height * 1.
    vs = len(verticesDict.keys())
    minX, minY = 0., 0.  # min(0., minCoord[1]), min(0., minCoord[0])
    maxX, maxY = w - 1., h - 1.  # min(w - 1, maxCoord[1]), min(h - 1, maxCoord[0])

    pointsBdary = [(minX, minY), (maxX, minY), (maxX, maxY), (minX, maxY)]  # xy coord
    pointsBdaryIdx = []
    newVsNum = 0
    for point in pointsBdary:
        if verticesDict.__contains__(point):
            pointsBdaryIdx.append(verticesDict[point][0])
        else:
            verticesDict[point] = [vs + newVsNum]
            pointsBdaryIdx.append(vs + newVsNum)
            newVsNum += 1
    pointsBdaryIdx.append(pointsBdaryIdx[0])
    for idx, point in enumerate(pointsBdary):
        verticesDict[point].append(pointsBdaryIdx[idx + 1])

    vertices, edgeIndex = isLineIntersection(verticesDict, scaleCoeff=s)
    walls = dict(vertices=vertices, segments=edgeIndex)
    ecldPVs = []
    for p in pDoors + pWindows:
        ecldPVs = ecldPVs + p
    walls, extendWalls = extendCornerWall(
        walls, min(maxX, maxY), scaleCoeff=s,
        excludeVIdxes=ecldPVs
    )

    plt.rcParams['savefig.dpi'] = 50
    plt.rcParams['figure.figsize'] = (w / 100., h / 100.)
    fig, ax = plt.subplots()
    myPlot(ax, **walls)
    drawPrimitives(ax, pDoors, vertices, color='black')
    drawPrimitives(ax, pWindows, vertices, color='blue')
    ax.axis('off')
    ax.margins(0.0)
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    fig.canvas.draw()
    wImg, hImg = fig.canvas.get_width_height()
    img = Image.frombytes('RGB', (wImg, hImg), fig.canvas.tostring_rgb())
    if wImg == width and hImg == height:
        pass
    else:
        img = img.resize((width, height), resample=Image.NEAREST)
    imgArray = np.array(img)
    transparency = np.ones((height, width, 1), dtype=np.uint8) * 255
    imgArray = np.concatenate([imgArray, transparency], axis=-1)

    if w >= h:
        scaleFactor = w / reSize
        wRe = 512
        hRe = int(h // scaleFactor)
        if hRe % 2 == 1:
            hRe = hRe + 1
    else:
        scaleFactor = h / reSize
        wRe = int(w // scaleFactor)
        hRe = 512
        if wRe % 2 == 1:
            wRe = wRe + 1
    fpLbl = lblImg.resize((wRe, hRe), resample=Image.NEAREST)
    lblArray = np.array(fpLbl)
    wPad = int((reSize - wRe) / 2)
    hPad = int((reSize - hRe) / 2)

    lblArray = np.pad(
        lblArray,
        ((hPad, hPad), (wPad, wPad)),
        'constant', constant_values=255
    )

    newLblArray = alterLabel(imgArray, np.array(lblImg))
    fpNewLbl = Image.fromarray(newLblArray, mode='P')

    palette = lblImg.getpalette()
    fpImg = Image.fromarray(imgArray, mode='RGBA')
    fpLbl = Image.fromarray(lblArray, mode='P')
    fpLbl.putpalette(palette)
    fpNewLbl.putpalette(palette)
    fpImg.save(os.path.join(imgSaveDir, imgName + '.png'))
    fpNewLbl.save(os.path.join(labelSaveDir, imgName + '.png'))
    plt.close()


if __name__ == "__main__":
    """
    数据对应关系；
    完整版，基于wall_svg.png和XXX.txt；

    目标：由于wall_svg.png中存在房间内的划分，导致歧义，因此按照geometric primitive的
    剖分规整标签，并保存在其它文件夹中
    """
    dataDir = "E:\\Datasets\\cubicasa5k"
    split = 'val'
    dataFile = "\\{}.txt".format(split)
    folders = np.genfromtxt(dataDir + dataFile, dtype='str')
    excludeList = [
        '\\high_quality_architectural\\2003\\',
        '\\high_quality_architectural\\2565\\',
        '\\high_quality_architectural\\6143\\',
        '\\high_quality_architectural\\10074\\',
        '\\high_quality_architectural\\10754\\',
        '\\high_quality_architectural\\10769\\',
        '\\high_quality_architectural\\14611\\',
        '\\high_quality\\7092\\',
        '\\high_quality\\1692\\',

        'high_quality_architectural\\10',  # img does not match label
    ]
    imgDir = '{}\\img_partition'.format(TARGET_DIR)
    labelDir = '{}\\annotation'.format(TARGET_DIR)

    if not os.path.exists(imgDir):
        os.makedirs(imgDir)
    if not os.path.exists(labelDir):
        os.makedirs(labelDir)

    ffolders = []
    for x in folders:
        x = x.replace('/', '\\')
        if x not in excludeList:
            ffolders.append(x)
    fs = [dataDir + x for x in ffolders]
    sorted(fs)

    """multi process"""
    # import multiprocessing
    # from multiprocessing import Pool
    # from tqdm import tqdm
    # from functools import partial
    #
    # cpuCount = multiprocessing.cpu_count() - 4
    # pool = Pool(cpuCount)
    #
    # func = partial(worker, imgSaveDir=imgDir, labelSaveDir=labelDir)
    # for res in tqdm(pool.imap_unordered(func, fs), total=len(fs)):
    #     pass

    """single process"""
    from tqdm import tqdm
    for f in tqdm(fs):
        worker(f, imgDir, labelDir)