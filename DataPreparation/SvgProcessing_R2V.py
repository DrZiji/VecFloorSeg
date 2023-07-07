import copy

import numpy as np
import cv2
import os
import triangle as tr
import matplotlib.pyplot as plt

from Utils import triPlot as trP
from DataPreparation.SvgProcessing_CubiCasa import clipBoundary, _genTriangleGraph, \
    buildDualRelationship
from skimage.draw import polygon
from PIL import Image
from Utils.graphicsUtilsRe import isLineIntersection, graphCrune
from Utils.extendWall import extendCornerWallR2V
from matplotlib.colors import ListedColormap

# abandoned
TARGET_DIR = 'jp_CAD'
CLASSES = (
"wall",
"kitchen",
"bathroom",
"bedroom",
"balcony",
"closet",
"hall",
"background",
"otherRoom"
)

def SVGParserR2V(filepath):
    THRESHOLD = 3
    """思路：
    （1）根据墙体、找到轮廓
    （2）对轮廓中的点进行合并
    （3）制作wall dict，处理方法和CUBICASA一致；
    （4）启发式延长线的画法：度为2的点就延长（因为数据中基本没有度为3的点）
    """
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[gray < 255] = 0
    tgt, contours, h = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    wallVertices = {}
    vIdx = 0
    for contour in contours:
        contour = contour.squeeze(1)
        new_contour = []
        for i in range(contour.shape[0]):
            # 暂时先这样，保留其中某一个点，这样做的不利因素是，延长的时候恐怕出现问题
            # if np.linalg.norm(contour[i] - contour[i - 1]) < THRESHOLD:
            #     pass
            # else:
            new_contour.append(contour[i])

        for i in range(len(new_contour)):
            v = (new_contour[i][0], new_contour[i][1])
            vPre = (new_contour[i - 1][0], new_contour[i - 1][1])
            if not wallVertices.__contains__(vPre):
                wallVertices[vPre] = [vIdx]
                vIdx += 1

            if not wallVertices.__contains__(v):
                wallVertices[v] = [vIdx, wallVertices[vPre][0]]
                vIdx += 1
            else:
                wallVertices[v].append(wallVertices[vPre][0])

    return wallVertices


def delaunayTriangulation(verticesDict, filePth, scaleCoeff=10, **kwargs):
    """
    执行三角剖分的API，其中的op是固定的，若想要调整，参考triangle API
    :param verticesDict: dict(key=户型图端点坐标，value=list(自身id，延申出去的墙体的另一个端点的ids))；
    :param filePth: 文件名
    :param scaleCoeff: 放大系数，当一些点的坐标距离很近时，计算可能出现误差，因此先将坐标放大；
    :param kwargs:
    :return:
    """
    EXTEND_THRESHOLD = 5
    EXTEND_THRESHOLD_1 = 20
    # lblImg = Image.open(filePth + 'wall_svg.png')
    fileName = filePth.split('\\')[-1].split('_')[0]
    split = filePth.split('\\')[-2]
    lblImg = Image.open(filePth)

    # 处理超出边界的端点，加上边界框
    width, height = lblImg.width, lblImg.height
    w, h = width * 1., height * 1.
    vs = len(verticesDict.keys())
    minX, minY = 0., 0.
    maxX, maxY = w - 1., h - 1.

    pointsBdary = [(minX, minY), (maxX, minY), (maxX, maxY), (minX, maxY)] # xy coord
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

    # 判断是否相交
    vertices, edgeIndex = isLineIntersection(verticesDict, scaleCoeff=scaleCoeff)
    walls = dict(vertices=vertices, segments=edgeIndex, **kwargs)
    walls, extendWalls = extendCornerWallR2V(
        walls, (EXTEND_THRESHOLD, EXTEND_THRESHOLD_1), scaleCoeff=scaleCoeff
    )

    # 记录第二次islineIntersection之前，walls中的顶点数量
    num_extend1_vs = len(walls['vertices'])
    vertices, edgeIndex = isLineIntersection(walls, scaleCoeff=1)
    # 之后vertices的数量如果有增多，说明是延长线之间的交点
    # 从延长线出发的交点，一定是延长线，属于extendWalls
    for vIdx in range(num_extend1_vs, len(vertices)):
        for edge in edgeIndex:
            # 这样加入edge会导致extendWalls重复，但是没有关系
            if edge[0] == vIdx or edge[1] == vIdx:
                extendWalls.append(edge)

    walls = dict(vertices=vertices, segments=edgeIndex)

    # fig, ax = plt.subplots()
    # trP.plot(ax, **walls)
    # fig.show()
    # plt.show()
    # plt.close()

    # delaunay三角化
    op = 'p'
    segWalls = tr.triangulate(walls, op)
    return walls, segWalls, extendWalls


def triangleCorrespondingLabel(segWalls, svgPth, scaleCoeff=1):
    """
    对每一个三角形所属的户型类别进行判定，统计三角形涵盖的所有像素所属类别，
    并进行投票，像素数量最多的类别对应该三角形的类别。
    :param segWalls: 进行delaunay三角剖分后的数据结构
    :param svgPth: 数据的名称
    :param scaleCoeff: 缩放比例
    :return: 向segWalls中加入triangles_label和cmap属性,
        shape(triangles_label) == shape(segWalls['triangles'])
    """
    # 注意这里的区别，不同的图像是不一样的label，
    # lblImg = Image.open(svgPth + 'wall_svg.png')
    fileName = svgPth.split('\\')[-1].split('_')[0]
    split = svgPth.split('\\')[-2]
    lblImg = Image.open('E:\\Datasets\\{}\\annotation\\{}\\{}.png'.format(TARGET_DIR, split, fileName))
    lblArray = np.array(lblImg, dtype=np.uint8)

    # # 初始化ndarray
    # 缩放到原始图像的尺寸
    h, w = lblArray.shape
    originVCoords = segWalls['vertices'] / float(scaleCoeff)
    originVCoords = np.around(originVCoords, 1).astype(int)

    # 计算label
    triangles = segWalls['triangles'].tolist()
    lblTris = np.zeros([len(triangles), 1], dtype=np.uint8)
    areaTris = np.zeros([len(triangles), 1], dtype=float)
    for idx, triangle in enumerate(triangles):
        triX, triY = originVCoords[triangle, 0], originVCoords[triangle, 1]
        area = 0.5 * np.abs(
            (triX[0] * triY[1] + triX[1] * triY[2] + triX[2] * triY[0]) - \
            (triX[1] * triY[0] + triX[2] * triY[1] + triX[0] * triY[2])
        )
        areaTris[idx] = area

        rr, cc = polygon(triY, triX) # shape(rr) == shape(cc)
        rrClip, ccClip = clipBoundary(rr, cc, h, w)
        lblTriCandidate = lblArray[rrClip, ccClip]
        if lblTriCandidate.shape[0] == 0:
            # 这个三角形面积在当前scaleCoeff太小了，将它的label设置为255(ignore)
            temp = 255
        else:
            lblTri, lblTriCounts = np.unique(lblTriCandidate, return_counts=True)
            tempIdx = np.argmax(lblTriCounts)
            temp = lblTri[tempIdx]

        lblTris[idx] = temp

    segWalls['triangles_label'] = lblTris
    segWalls['triangles_area'] = areaTris
    assert 0.95 < np.sum(areaTris) / (h * w) < 1.05, "wrong triangle area calculation."

    palette = lblImg.getpalette()
    palette = np.array(palette).reshape((256, 3)) / 256.
    paletteTransparency = np.ones((256, 1))
    palette = np.concatenate([palette, paletteTransparency], axis=-1)

    segWalls['cmap'] = ListedColormap(palette)
    return segWalls


def plotMergeGraph(
        walls, segWalls, topoDict,
        filePth, saveDir, plot_label=True
):
    from Utils.triPlot import plot
    trP.compare(
        plt, walls, segWalls,
        figsize=(12, 10), plot_label=plot_label
    )
    ax3 = plt.subplot(236)

    img = plt.imread(filePth)
    ax3.imshow(img)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)

    ax = plt.subplot(235)
    plot(ax, **topoDict)
    plt.savefig(os.path.join(saveDir, filePth.split('\\')[-1].split('_')[0] + '_merge.png'))
    plt.close()


def plotTriangles(walls, segWalls, mergeDict, filePth, saveDir, plot_label=True):
    """
    数据的可视化，对比生成结果和原始结果
    :param walls: 墙体拓扑
    :param segWalls:
    :param filePth:
    :return:
    """
    def merge2Tri(primitives_merge, lbls_merge, primitives_triangle):

        lbls_triangle = np.ones([primitives_triangle.shape[0], 1]) * 255
        for m, m_tris in primitives_merge.items():
            mLabel = lbls_merge[m]
            lbls_triangle[m_tris] = mLabel[0]

        return lbls_triangle

    convert_lbls_triangle = merge2Tri(
        mergeDict['merge2tri'], mergeDict['merge_label'],
        segWalls['triangles_label']
    )
    # 对segWalls中的结果进行调整，目的是可视化merge之后的label是什么样的
    newSegWalls = copy.deepcopy(segWalls)
    newSegWalls['triangles_label'] = convert_lbls_triangle

    trP.compare(plt, walls, newSegWalls, figsize=(12, 10), plot_label=plot_label)
    ax3 = plt.subplot(236)

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(os.path.join(saveDir, filePth.split('\\')[-1]))
    plt.close()

def worker(imgPth, dataVerifyDir=None):
    # 注意:在处理CubiCasa数据时,scaleCoeff设置为50
    s = 1

    vDict = SVGParserR2V(imgPth)

    walls, segWalls, extendWalls = delaunayTriangulation(
        vDict, imgPth, scaleCoeff=s
    )
    segWalls = triangleCorrespondingLabel(segWalls, imgPth, scaleCoeff=s)

    fWall = []
    for wall in segWalls['segments']:
        w = (wall[0], wall[1]) if wall[0] < wall[1] else (wall[1], wall[0])
        fWall.append(w)

    tEdges, tEdgeAttr, tEdgeType, nodes, nodeAttr, tEdgeDual, wholeEdgeSet = \
        _genTriangleGraph(segWalls['vertices'], segWalls['triangles'], fWall, extendWalls, [], [])

    fplanVeno = {
        'vertices': nodes, 'x': nodeAttr,
        'segments': tEdges, 'segment_attr': tEdgeAttr,
        'segments_type': tEdgeType, 'scale_coeff': s,
        'segment_dual': tEdgeDual
    }
    segWalls = buildDualRelationship(
        segWalls=segWalls, venoGraph=fplanVeno, edgeTriPairs=wholeEdgeSet
    )

    merge_segWalls, merge_fplanVeno = graphCrune(segWalls, fplanVeno, walls, wholeEdgeSet)

    if dataVerifyDir is not None:
        mergeDict = {
            'merge2tri': merge_fplanVeno['merge2tri'],
            'merge_label': merge_segWalls['triangles_label'],
        }

        # 检查edge attr是否标注错误
        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
        # debug_vertices = segWalls['vertices']
        # debug_segments_attr = np.array(segWalls['edge_attr'])
        # debug_segments = np.array(segWalls['edge'])
        # debug_wall_segments = debug_segments[debug_segments_attr[:, 0] == 0]
        # debug_partition_segments = debug_segments[debug_segments_attr[:, 1] == 0]
        #
        # topo =  {
        #     'vertices': debug_vertices,
        #     'segments': list(debug_partition_segments)
        # }
        # axes.set_title('wall topo')
        # trP.plot(axes, **topo)
        # fig.savefig(
        #     os.path.join(dataVerifyDir, imgPth.split('\\')[-1])
        # )
        # plt.close()
        plotTriangles(walls, segWalls, mergeDict, imgPth, dataVerifyDir)
        topoDict = {
            # 'vertices': merge_fplanVeno['vertices'],
            # 'segments': merge_fplanVeno['segments']
            # 测试边界点是不是连接到了primitive的中心点上
            'vertices': merge_segWalls['vertices'],
            'segments': merge_segWalls['inner2bd_index'],
        }
        plotMergeGraph(
            walls, merge_segWalls, topoDict,
            imgPth, dataVerifyDir, plot_label=False
        )

    return imgPth.split('\\')[-1].split('_')[0], segWalls, fplanVeno, merge_segWalls, merge_fplanVeno


if __name__ == "__main__":

    dataDir = "E:\\Datasets\\{}\\ann_dir".format(TARGET_DIR)
    srcDir = "E:\\Datasets\\jp"
    split = 'test'
    dataVerificationDir = '..\\ConvexHull\\merge_R2V' # '..\\ConvexHull\\merge_R2V'
    ffolders = os.listdir(os.path.join(dataDir, split))

    fs = [os.path.join(srcDir, split, x.split('.')[0] + '_close_wall.png') for x in ffolders]
    sorted(fs)

    import multiprocessing
    from multiprocessing import Pool
    from tqdm import tqdm
    import pickle

    cpuCount = multiprocessing.cpu_count() - 4
    pool = Pool(cpuCount)

    writeIn = {}
    merge_writeIn = {}
    if dataVerificationDir is not None:
        import functools
        func = functools.partial(worker, dataVerifyDir=dataVerificationDir)
    else:
        func = worker
    for res in tqdm(pool.imap_unordered(func, fs), total=len(fs)):
        writeIn[res[0]] = [res[1], res[2]]
        merge_writeIn[res[0]] = [res[3], res[4]]

    # with open('E:\\Datasets\\{}\\{}_phase_V4.pkl'.format(TARGET_DIR, split), 'wb') as f:
    #     pickle.dump(writeIn, f)
    # with open('E:\\Datasets\\{}\\merge_{}_phase_V4.pkl'.format(TARGET_DIR, split), 'wb') as f:
    #     pickle.dump(merge_writeIn, f)

    # from tqdm import tqdm
    #
    # for f in tqdm(fs[:]):
    #     worker(f, dataVerificationDir)

