import copy
import queue

import numpy as np

from DataPreparation import PARTITION, NOTPARTITION, NOTWALL
from collections import deque, Counter
from functools import cmp_to_key

OVERLAPPING = 'overlapping'
ENDPOINTS_EDGE = 'endpoints of edge'
ENDPOINTS_AEDGE = 'endpoints of another edge'
INTERSECTION = 'intersection'


def isLineIntersection(verticesDict, scaleCoeff=1):

    if len(verticesDict.keys()) == 2:
        vertices = verticesDict['vertices']
        edges = verticesDict['segments']
    else:

        vertices = np.array(list(verticesDict.keys()))
        vertices = np.around(vertices, 0) * scaleCoeff
        vertices = vertices.tolist()
        vertices = [tuple(v) for v in vertices]

        edgeMatrix = np.zeros((len(vertices), len(vertices)), dtype=np.uint8)
        for v in verticesDict.values():
            if len(v) == 1:
                pass
            else:
                start = v[0]
                for x in v[1:]:
                    if x > start:
                        v1, v2 = start, x
                    elif x < start:
                        v1, v2 = x, start
                    else:
                        continue
                    edgeMatrix[v1, v2] += 1
        edges = np.argwhere(edgeMatrix >= 1).tolist()
        edges = [tuple(e) for e in edges]

    oldEdges = deque(edges)
    newEdges = deque()

    while len(oldEdges) > 0:
        edge = oldEdges.popleft()
        e1St, e1Ed = edge
        e1StXY, e1EdXY = vertices[e1St], vertices[e1Ed]
        newVertices = []
        crossSegments = []
        homoSegments = []
        verticesOnLine = []
        rmSegments = []

        for anotherEdge in oldEdges:
            e2St, e2Ed = anotherEdge
            e2StXY, e2EdXY = vertices[e2St], vertices[e2Ed]
            if e1St == e2St or e1St == e2Ed or e1Ed == e2St or e1Ed == e2Ed:
                # 交点有相交，判断是否两个线段是否有重合
                if judgeParallel(e1StXY, e1EdXY, e2StXY, e2EdXY):
                    if (e2St, e2StXY) not in verticesOnLine:
                        verticesOnLine.append((e2St, e2StXY))
                    if (e2Ed, e2EdXY) not in verticesOnLine:
                        verticesOnLine.append((e2Ed, e2EdXY))
                    rmSegments.append(anotherEdge)

            else:
                intersection = judge(e1StXY, e1EdXY, e2StXY, e2EdXY)

                if intersection[0] == INTERSECTION: # 相交，交点非线段端点
                    v = (intersection[1], intersection[2])
                    if v not in vertices:
                        vertices.append(v)
                        intersectionIdx = len(vertices) - 1
                    else:
                        intersectionIdx = vertices.index(v)

                    newVertices.append((intersectionIdx, v, e2St, e2Ed))
                    rmSegments.append(anotherEdge) # 将相交的直线按照交点拆分开

                elif intersection[0] == OVERLAPPING: # 重合
                    verticesOnLine.extend([(e2St, e2StXY), (e2Ed, e2EdXY)])
                    rmSegments.append(anotherEdge)

                elif intersection[0] == ENDPOINTS_EDGE: # 相交点是edge的端点，another edge被分为两段
                    temp1 = e1St if intersection[1] == 0 else e1Ed
                    tempV1 = (temp1, e2St) if e2St > temp1 else (e2St, temp1)
                    tempV2 = (temp1, e2Ed) if e2Ed > temp1 else (e2Ed, temp1)
                    rmSegments.append(anotherEdge)
                    crossSegments.extend([tempV1, tempV2])

                elif intersection[0] == ENDPOINTS_AEDGE:  # 相交点是another edge的端点, edge被分为两段
                    temp1 = (e2St, e2StXY) if intersection[1] == 0 else (e2Ed, e2EdXY)
                    verticesOnLine.append(temp1)
                else:
                    pass

        verticesOnLine.extend(newVertices)
        if len(verticesOnLine) > 0 or len(crossSegments) > 0:
            if (e1St, e1StXY) not in verticesOnLine:
                verticesOnLine.insert(0, (e1St, e1StXY))
            if (e1Ed, e1EdXY) not in verticesOnLine:
                verticesOnLine.insert(0, (e1Ed, e1EdXY))

            vsOL, vsOL_dict = [], {}
            for v in verticesOnLine:
                if not vsOL_dict.__contains__(v[0]):
                    vsOL_dict[v[0]] = list(v[1:])
                else:
                    if len(v) > 2:
                        vsOL_dict[v[0]].extend(v[2:])
            for k, v, in vsOL_dict.items():
                vsOL.append((k, *tuple(v)))

            sortedVs = sorted(vsOL, key=cmp_to_key(coordCmp))
            for newVIdx in range(1, len(sortedVs)):
                temp1 = sortedVs[newVIdx][0]
                temp2 = sortedVs[newVIdx - 1][0]
                if temp2 > temp1:
                    segment = (temp1, temp2)
                    homoSegments.append(segment)
                elif temp2 == temp1:
                    raise RuntimeWarning('there is still circumstances of points overlapping in sortedVs')
                else:
                    segment = (temp2, temp1)
                    homoSegments.append(segment)

                if len(sortedVs[newVIdx]) != 2:
                    temp1 = sortedVs[newVIdx][0]
                    lenPairs = (len(sortedVs[newVIdx]) - 2) / 2
                    for p in range(int(lenPairs)):
                        temp2, temp3 = sortedVs[newVIdx][2 + p * 2], \
                                        sortedVs[newVIdx][3 + p * 2]
                        tempV1 = (temp1, temp2) if temp2 > temp1 else (temp2, temp1)
                        tempV2 = (temp1, temp3) if temp3 > temp1 else (temp3, temp1)
                        crossSegments.extend([tempV1, tempV2])

            stIdx, edIdx = sortedVs.index((e1St, e1StXY)), sortedVs.index((e1Ed, e1EdXY))
            if stIdx > edIdx: big, small = stIdx, edIdx
            else: big, small = edIdx, stIdx
            rmSegments.extend(homoSegments[small: big])
            oldEdges.extend(homoSegments)
            oldEdges.extend(crossSegments)
            oldEdges = deque(list(set(oldEdges).difference(set(rmSegments))))
            newEdges.extend(homoSegments[small: big])
        else:
            newEdges.append(edge)

    finalEdges = {}
    for edge in newEdges:
        if not finalEdges.__contains__(edge):
            finalEdges[edge] = 0
        else:
            finalEdges[edge] += 1

    return vertices, list(finalEdges.keys())


def judge(A1, A2, B1, B2):
    Ax, Ay = A1
    Bx, By = A2
    Cx, Cy = B1
    Dx, Dy = B2

    if max(Ax, Bx) >= min(Cx, Dx) \
        and min(Ax, Bx) <= max(Cx, Dx) \
        and max(Ay, By) >= min(Cy, Dy) \
        and min(Ay, By) <= max(Cy, Dy):

        crossAB_CD = (Bx - Ax) * (Dy - Cy) - (By - Ay) * (Dx - Cx)
        if crossAB_CD != 0:
            coeffLambda = ((Cx - Ax) * (Dy - Cy) - (Cy - Ay) * (Dx - Cx)) / crossAB_CD
            coeffMu = ((Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)) / (crossAB_CD * -1)

            if 0 < coeffMu < 1 and 0 < coeffLambda < 1:
                crossx = Ax + coeffLambda * (Bx - Ax)
                crossy = Ay + coeffLambda * (By - Ay)

                crossx1 = Cx + coeffMu * (Dx - Cx)
                crossy1 = Cy + coeffMu * (Dy - Cy)

                assert abs(crossx1 - crossx) < 0.2 and abs(crossy1 - crossy) < 0.2, \
                    "the wrong intersection point calculated, notice!!!"
                fX, fY = round(crossx) * 1., round(crossy) * 1.

                if (fX, fY) == A1:
                    return ENDPOINTS_EDGE, 0
                if (fX, fY) == A2:
                    return ENDPOINTS_EDGE, 1
                if (fX, fY) == B1:
                    return ENDPOINTS_AEDGE, 0
                if (fX, fY) == B2:
                    return ENDPOINTS_AEDGE, 1

                return INTERSECTION, fX, fY

            elif (coeffLambda == 0 or coeffLambda == 1) and 0 < coeffMu < 1:
                return ENDPOINTS_EDGE, coeffLambda

            elif (coeffMu == 0 or coeffMu == 1) and 0 < coeffLambda < 1:
                return ENDPOINTS_AEDGE, coeffMu

            else:
                return False, -1
        else:
            if (Cx - Ax) * (Dy - Cy) - (Cy - Ay) * (Dx - Cx) != 0:
                return False, -1
            else:
                return OVERLAPPING, -1
    else:
        return False, -1


def judgeParallel(A1, A2, B1, B2):
    if A1 == B1:
        coPoints, line1Another, line2Another = A1, A2, B2
    elif A2 == B1:
        coPoints, line1Another, line2Another = A2, A1, B2
    elif A1 == B2:
        coPoints, line1Another, line2Another = A1, A2, B1
    elif A2 == B2:
        coPoints, line1Another, line2Another = A2, A1, B1
    else:
        raise NotImplementedError

    line1 = (coPoints[0] - line1Another[0], coPoints[1] - line1Another[1])
    line2 = (coPoints[0] - line2Another[0], coPoints[1] - line2Another[1])

    cdot = line1[0] * line2[0] + line1[1] * line2[1]
    l = (line1[0] * line1[0] + line1[1] * line1[1]) * \
                (line2[0] * line2[0] + line2[1] * line2[1])
    if pow(cdot, 2) == l and cdot > 0:
        return True

    else:
        return False


def coordCmp(pt1, pt2):
    if pt1[1][0] > pt2[1][0]:
        return 1
    elif pt1[1][0] == pt2[1][0]:
        if pt1[1][1] > pt2[1][1]:
            return 1
        elif pt1[1][1] == pt2[1][1]:
            return 0
        else:
            return -1
    else:
        return -1


def graphCrune(oriDict, venoDict, originalWall, wholeEdgeSet):

    ORIGINAL_WALL = 1
    DELAUNAY_WALL = 0
    IGNORE_LABEL = 255
    def regionGrow():
        """
        :return: dict {group_id, [tri_idx]}
        """
        temp_ET_Pairs = copy.deepcopy(ET_Pairs)
        groups, group_id = {}, -1
        reserve_tris = []

        while sum(T_Counts) > 0:
            if len(reserve_tris) == 0:
                group_id += 1
                groups[group_id] = []
                for idx, i in enumerate(T_Counts):
                    if i == 1:
                        reserve_tris.append(idx)
                        break
                    else:
                        pass

            tri_id = reserve_tris.pop(0)
            T_Counts[tri_id] = 0
            groups[group_id].append(tri_id)
            edges = TE_Pairs[tri_id]
            for e in edges:
                e_mapping = temp_ET_Pairs[e]
                if e_mapping[-1] == ORIGINAL_WALL or e_mapping[1] == -1 or e_mapping[2] == -1:
                    pass
                else:
                    new_tri_id = e_mapping[2] if e_mapping[1] == tri_id else e_mapping[1]

                    if new_tri_id not in reserve_tris:
                        reserve_tris.append(new_tri_id)
                    e_mapping[-1] = ORIGINAL_WALL

        return groups


    T_Counts = [1] * len(venoDict['vertices'])
    ET_Pairs = {}
    TE_Pairs = {}
    for k, v in wholeEdgeSet.items():

        tri_idxes = [v[1], v[2]]
        for tri_idx in tri_idxes:
            if tri_idx == -1:
                continue
            if TE_Pairs.__contains__(tri_idx):
                TE_Pairs[tri_idx].append(k)
            else:
                TE_Pairs[tri_idx] = [k]

        v_copy = copy.deepcopy(v)
        if k in originalWall['segments']:
            v_copy.append(ORIGINAL_WALL)
        else:
            v_copy.append(DELAUNAY_WALL)
        ET_Pairs[k] = v_copy

    groups = regionGrow()

    new_venoGraph_vs = np.empty([len(groups.keys()), 2])
    new_oriGraph_tris_label = np.empty([len(groups.keys()), 1])
    new_oriGraph_tris_area = np.empty([len(groups.keys()), 1])

    new_ET_Pairs = {}

    for g_id, g_tri_idxes in groups.items():
        vs = np.array(venoDict['vertices'])
        g_inner_v = np.mean(vs[g_tri_idxes], axis=0)
        new_venoGraph_vs[g_id, :] = g_inner_v.tolist()

        vs_areas = oriDict['triangles_area'][g_tri_idxes]
        vs_area = np.sum(vs_areas)
        new_oriGraph_tris_area[g_id] = vs_area

        vs_label = oriDict['triangles_label'][g_tri_idxes]
        mask = vs_label < IGNORE_LABEL
        temp = vs_label[mask]
        if temp.shape[0] > 0:

            temp_idxes, temp_counts = np.unique(temp, return_counts=True)
            max_area = 0
            for temp_idx in temp_idxes:
                temp_area = vs_areas[vs_label == temp_idx].sum()
                if temp_area >= max_area:
                    max_idx = temp_idx
                    max_area = temp_area
            new_oriGraph_tris_label[g_id] = max_idx

        else:
            new_oriGraph_tris_label[g_id] = IGNORE_LABEL

        for g_tri_idx in g_tri_idxes:
            tri_edges = TE_Pairs[g_tri_idx]
            for tri_edge in tri_edges:

                if ET_Pairs[tri_edge][-1] != ORIGINAL_WALL:
                    continue
                edge_attrs = ET_Pairs[tri_edge]
                if new_ET_Pairs.__contains__(tri_edge):
                    new_ET_Pairs[tri_edge].append(g_id)
                else:
                    new_ET_Pairs[tri_edge] = [edge_attrs[0], g_id]

    new_O2V, new_V2O = {}, {}
    for new_e, new_v in new_ET_Pairs.items():
        assert len(new_v) <= 3, 'wrong edge polygon mapping.'
        if len(new_v) == 2:
            veno_v1 = new_v[-1]
            venoE = (veno_v1, -1)
            new_v.extend(
                [-1, ET_Pairs[new_e][-2]]
            )

        else:
            veno_v1, veno_v2 = new_v[-2], new_v[-1]
            if veno_v1 < veno_v2:
                venoE = (veno_v1, veno_v2)
            elif veno_v1 == veno_v2:
                venoE = (veno_v1, -1)
            else:
                venoE = (veno_v2, veno_v1)

            if new_oriGraph_tris_label[veno_v1] == new_oriGraph_tris_label[veno_v2]:
                new_v.append(NOTPARTITION)
            else:
                new_v.append(PARTITION)

        new_O2V[new_e] = venoE
        if venoE[-1] != -1:

            if new_V2O.__contains__(venoE):
                new_V2O[venoE].append(new_e)
            else:
                new_V2O[venoE] = [new_e]

    new_oriGraph_vs = {}
    new_oriGraph_es, new_oriGraph_es_attr, new_oriGraph_es_dual = [], [], []
    for k, v in new_ET_Pairs.items():
        if not new_oriGraph_vs.__contains__(k[0]):
            new_oriGraph_vs[k[0]] = 1
        if not new_oriGraph_vs.__contains__(k[1]):
            new_oriGraph_vs[k[1]] = 1

        new_oriGraph_es.append(k)
        new_oriGraph_es_attr.append((v[0], v[-1]))
        new_oriGraph_es_dual.append(new_O2V[k])

    new_venoGraph_segments = []
    new_venoGraph_segments_attr = []
    new_venoGraph_segments_type = []
    new_venoGraph_segment_dual = []
    for k, v in new_V2O.items():
        new_venoGraph_segments.append(k)
        if len(v) > 1:
            final_dual = v[0]
            for sub_v in v:
                if new_ET_Pairs[v[0]][-1] == PARTITION:
                    final_dual = sub_v
                    break

            new_venoGraph_segment_dual.append(final_dual)
            new_venoGraph_segments_type.append(new_ET_Pairs[final_dual][0])
        else:
            new_venoGraph_segment_dual.append(v[0])
            new_venoGraph_segments_type.append(new_ET_Pairs[v[0]][0])

        seg_attr1, seg_attr2 = new_venoGraph_vs[k[0]], new_venoGraph_vs[k[1]]
        seg_attr = seg_attr1.tolist() + seg_attr2.tolist()
        new_venoGraph_segments_attr.append(seg_attr)

    assert len(
        set(new_venoGraph_segment_dual).difference(set(new_oriGraph_es))
    ) == 0, 'attention'

    venoVs = np.array(venoDict['vertices'])
    new_oriGraph_vs = np.concatenate(
        [oriDict['vertices'], venoVs, new_venoGraph_vs], axis=0
    )

    num_old_oriGraph_vs = oriDict['vertices'].shape[0]
    num_old_venoGraph_vs = venoVs.shape[0]
    ori_triangles = oriDict['triangles']
    edges_merge2bd = []
    for k, v in groups.items():
        tri_vIdxes = ori_triangles[v].reshape(-1).tolist()
        tri_vIdxes = list(set(tri_vIdxes))
        edges_merge2bd.extend([
            (t, k + num_old_oriGraph_vs + num_old_venoGraph_vs) for t in tri_vIdxes
        ])
        edges_merge2bd.extend([
            (t + num_old_oriGraph_vs, k + num_old_oriGraph_vs + num_old_venoGraph_vs) for t in v
        ])


    new_oriDict = {
        'vertices': new_oriGraph_vs,
        'segments': new_oriGraph_es,
        'triangles_label': new_oriGraph_tris_label,
        'triangles_area': new_oriGraph_tris_area,
        'edge': new_oriGraph_es,
        'edge_attr': new_oriGraph_es_attr,
        'edge_dual': new_oriGraph_es_dual,
        'inner2bd_index': edges_merge2bd,
    }

    new_venoDict = {
        'vertices': new_venoGraph_vs,
        'x': new_venoGraph_vs,
        'segments': new_venoGraph_segments,
        'segment_attr': new_venoGraph_segments_attr,
        'segments_type': new_venoGraph_segments_type,
        'segments_dual': new_venoGraph_segment_dual,
        'scale_coeff': venoDict['scale_coeff'],
        'merge2tri': groups,
    }
    return new_oriDict, new_venoDict