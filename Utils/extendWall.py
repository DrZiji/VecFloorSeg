import numpy as np

RATIO_THRESHOLD = 0.8


def extendCornerWall(newWallDict, DIST_THRESHOLD, scaleCoeff=1, excludeVIdxes=None):

    CORNER_LOWEST_BOUND = -0.95
    CORNER_LOWER_BOUND = -0.05
    CORNER_HIGHER_BOUND = 0.05
    EXTEND_UPPER_BOUND = 3


    EXTEND_TYPE = ['corner', 'link']

    vs = newWallDict['vertices']
    segs = newWallDict['segments']

    segsArray = np.array(segs)
    aSegsArray = segsArray[:, [1, 0]].copy()
    segsArray = np.concatenate([segsArray, aSegsArray], axis=0)

    vDegreeDict = dict()
    for i in range(len(segsArray)):
        vIdx = segsArray[i, 0]
        if not vDegreeDict.__contains__(vIdx):
            vDegreeDict[vIdx] = [1, segsArray[i, 1]]
        else:
            vDegreeDict[vIdx][0] += 1
            vDegreeDict[vIdx].append(segsArray[i, 1])

    candiVIdxes, candiCorners = [], []
    for k, v in vDegreeDict.items():
        if k in excludeVIdxes:
            continue
        if v[0] == 2:
            candiVIdxes.append(k)

        elif v[0] == 3:
            temp = (k, *v[1:])
            coordC_V = np.array([vs[x] for x in temp]) # 4,2
            l01, l02, l03 = coordC_V[1] - coordC_V[0], \
                            coordC_V[2] - coordC_V[0], \
                            coordC_V[3] - coordC_V[0]
            len_l01, len_l02, len_l03 = np.linalg.norm(l01), \
                                        np.linalg.norm(l02), \
                                        np.linalg.norm(l03)
            c01_02, c01_03, c02_03 = np.dot(l01, l02) / (len_l01 * len_l02 + 1e-7), \
                                     np.dot(l01, l03) / (len_l01 * len_l03 + 1e-7), \
                                     np.dot(l02, l03) / (len_l02 * len_l03 + 1e-7),

            if min(c01_02, c01_03, c02_03) < CORNER_LOWEST_BOUND:
                if c01_02 < CORNER_LOWEST_BOUND:
                    target = (k, *v[1:], EXTEND_TYPE[1])
                elif c01_03 < CORNER_LOWEST_BOUND:
                    target = (k, v[1], v[3], v[2], EXTEND_TYPE[1])
                elif c02_03 < CORNER_LOWEST_BOUND:
                    target = (k, v[2], v[3], v[1], EXTEND_TYPE[1])
                else:
                    target = None
            else:
                if CORNER_LOWER_BOUND < c01_02 < CORNER_HIGHER_BOUND:
                    # 01_02对应的夹角是90度
                    corner1, corner2, link = l01, l02, l03
                    temp1 = (k, *v[1:])
                elif CORNER_LOWER_BOUND < c01_03 < CORNER_HIGHER_BOUND:
                    corner1, corner2, link = l01, l03, l02
                    temp1 = (k, v[1], v[3], v[2])
                elif CORNER_LOWER_BOUND < c02_03 < CORNER_HIGHER_BOUND:
                    corner1, corner2, link = l02, l03, l01
                    temp1 = (k, v[2], v[3], v[1])
                else:
                    continue

                if np.dot(corner1 + corner2, link) > 0:
                    target = (*temp1, EXTEND_TYPE[0])
                else:
                    target = None

            if isinstance(target, tuple):
                candiCorners.append(target)

    eEdgeDict, extendEdges = {}, []
    for c_vIdx in candiVIdxes:
        coordC_V = np.array(vs[c_vIdx])
        neibour = vDegreeDict[c_vIdx]
        for i, vIdx in enumerate(neibour[1:]): # i=0,1
            if True:
                ano_i = len(neibour) - 1 - i
                ano_vIdx = neibour[ano_i]
                coordV = np.array(vs[vIdx])
                coord_anoV = np.array(vs[ano_vIdx])

                dist = np.linalg.norm(coordV - coordC_V)
                dist_ano = np.linalg.norm(coord_anoV - coordC_V)
                dist_ratio = dist / dist_ano

                if True:
                    key = (vIdx, c_vIdx) if c_vIdx > vIdx else (c_vIdx, vIdx)
                    value = (-1, ano_vIdx) if c_vIdx > vIdx else (ano_vIdx, -1)
                    if eEdgeDict.__contains__(key):
                        eEdgeDict[key] = (eEdgeDict[key][0] + value[0] + 1, eEdgeDict[key][1] + value[1] + 1)
                    else:
                        eEdgeDict[key] = value
                    break


    for k, v in eEdgeDict.items():
        for kelem, velem in zip(k, v):
            if velem == -1:
                continue
            else:
                extendEdges.append((kelem, velem))
        extendEdges.append(k)
        extendEdges.append((k[1], k[0]))

    for candiV in candiCorners:
        if candiV[-1] == EXTEND_TYPE[0]:
            l1 = (candiV[0], candiV[1])
            l2 = (candiV[0], candiV[2])
            extendEdges.extend([l1, l2])

        elif candiV[-1] == EXTEND_TYPE[1]:
            l = (candiV[0], candiV[3])
            extendEdges.append(l)

        else:
            raise RuntimeError('Only extend corner or edge.')

    newSegs = []
    for extendE in extendEdges:
        A1, A2 = vs[extendE[1]], vs[extendE[0]] # XY
        minLambda = 1e7
        crossX, crossY = -1, -1
        for seg in segs:
            if seg == extendE or (seg[1], seg[0]) == extendE:
                continue
            B1, B2 = vs[seg[0]], vs[seg[1]]
            insec_stat, X, Y, coeffLambda = calInsecPt(A1, A2, B1, B2)
            if insec_stat and coeffLambda < minLambda:
                minLambda = coeffLambda
                crossX = X
                crossY = Y
            else:
                pass

        if crossX == -1 and crossY == -1:
            continue

        extVec = np.array([crossX - A2[0], crossY - A2[1]])
        minExtDist = np.sqrt(np.dot(extVec, extVec))
        # if minExtDist > DIST_THRESHOLD * EXTEND_UPPER_BOUND * 8 and minLambda > EXTEND_UPPER_BOUND:
        #     crossX = (A2[0] - A1[0]) * EXTEND_UPPER_BOUND + A1[0]
        #     crossY = (A2[1] - A1[1]) * EXTEND_UPPER_BOUND + A1[1]

        if (crossX, crossY) in vs:
            tarIdx = vs.index((crossX, crossY))
        else:
            vs.append((crossX, crossY))
            tarIdx = len(vs) - 1
        edge = (extendE[0], tarIdx) if extendE[0] < tarIdx else (tarIdx, extendE[0])
        newSegs.append(edge)

    segs.extend(newSegs)
    newWalls = dict(vertices=vs, segments=segs)
    return newWalls, newSegs


def extendFloatingWall(newWallDict, DIST_THRESHOLD, scaleCoeff=1):

    vs = newWallDict['vertices']
    segs = newWallDict['segments']

    segsArray = np.array(segs)
    aSegsArray = segsArray[:, [1, 0]].copy()
    segsArray = np.concatenate([segsArray, aSegsArray], axis=0)

    vDegreeDict = dict()
    for i in range(len(segsArray)):
        vIdx = segsArray[i, 0]
        if not vDegreeDict.__contains__(vIdx):
            vDegreeDict[vIdx] = [1, segsArray[i, 1]]
        else:
            vDegreeDict[vIdx][0] += 1
            vDegreeDict[vIdx].append(segsArray[i, 1])

    candiVIdxes, eEdgeDict, extendEdges, newSegs = [], {}, [], []
    for k, v in vDegreeDict.items():
        if v[0] == 2:
            candiVIdxes.append(k)

    for c_vIdx in candiVIdxes:
        coordC_V = np.array(vs[c_vIdx])
        neibour = vDegreeDict[c_vIdx]
        for i, vIdx in enumerate(neibour[1:]): # i=0,1
            if vDegreeDict[vIdx][0] == 2:
                ano_i = len(neibour) - 1 - i
                ano_vIdx = neibour[ano_i]
                coordV = np.array(vs[vIdx])
                coord_anoV = np.array(vs[ano_vIdx])

                dist = np.linalg.norm(coordV - coordC_V)
                dist_ano = np.linalg.norm(coord_anoV - coordC_V)
                dist_ratio = dist / dist_ano

                if True:
                    key = (vIdx, c_vIdx) if c_vIdx > vIdx else (c_vIdx, vIdx)
                    value = (-1, ano_vIdx) if c_vIdx > vIdx else (ano_vIdx, -1)
                    if eEdgeDict.__contains__(key):
                        eEdgeDict[key] = (eEdgeDict[key][0] + value[0] + 1, eEdgeDict[key][1] + value[1] + 1)
                    else:
                        eEdgeDict[key] = value
                    break

    for k, v in eEdgeDict.items():
        for kelem, velem in zip(k, v):
            if velem == -1:
                continue
            else:
                extendEdges.append((kelem, velem))
        extendEdges.append(k)
        extendEdges.append((k[1], k[0]))

    for extendE in extendEdges:
        A1, A2 = vs[extendE[1]], vs[extendE[0]]
        minLambda = 1e7
        crossX, crossY = 0, 0
        for seg in segs:
            if seg == extendE or (seg[1], seg[0]) == extendE:
                continue
            B1, B2 = vs[seg[0]], vs[seg[1]]
            insec_stat, X, Y, coeffLambda = calInsecPt(A1, A2, B1, B2)
            if insec_stat and coeffLambda < minLambda:
                minLambda = coeffLambda
                crossX = X
                crossY = Y
            else:
                pass

        if (crossX, crossY) in vs:
            tarIdx = vs.index((crossX, crossY))
        else:
            vs.append((crossX, crossY))
            tarIdx = len(vs) - 1
        edge = (extendE[0], tarIdx) if extendE[0] < tarIdx else (tarIdx, extendE[0])
        segs.append(edge)
        newSegs.append(edge)

    newWalls = dict(vertices=vs, segments=segs)
    return newWalls, newSegs


def extendCornerWallR2V(newWallDict, DIST_THRESHOLD, scaleCoeff=1):

    CORNER_LOWEST_BOUND = -0.95
    CORNER_LOWER_BOUND = -0.05
    CORNER_HIGHER_BOUND = 0.05
    EXTEND_UPPER_BOUND = 3


    EXTEND_TYPE = ['corner', 'link']

    vs = newWallDict['vertices']
    segs = newWallDict['segments']

    segsArray = np.array(segs)
    aSegsArray = segsArray[:, [1, 0]].copy()
    segsArray = np.concatenate([segsArray, aSegsArray], axis=0)

    vDegreeDict = dict()
    for i in range(len(segsArray)):
        vIdx = segsArray[i, 0]
        if not vDegreeDict.__contains__(vIdx):
            vDegreeDict[vIdx] = [1, segsArray[i, 1]]
        else:
            vDegreeDict[vIdx][0] += 1
            vDegreeDict[vIdx].append(segsArray[i, 1])

    candiVIdxes, candiCorners = [], []
    for k, v in vDegreeDict.items():
        if v[0] == 2:
            candiVIdxes.append(k)

    eEdgeDict, extendEdges = {}, []
    for c_vIdx in candiVIdxes:
        coordC_V = np.array(vs[c_vIdx])
        neibour = vDegreeDict[c_vIdx]
        for i, vIdx in enumerate(neibour[1:]):
            ano_i = len(neibour) - 1 - i
            ano_vIdx = neibour[ano_i]
            coordV = np.array(vs[vIdx])
            coord_anoV = np.array(vs[ano_vIdx])

            dist = np.linalg.norm(coordV - coordC_V)
            dist_ano = np.linalg.norm(coord_anoV - coordC_V)

            if dist < DIST_THRESHOLD[0]:
                key = (vIdx, c_vIdx) if c_vIdx > vIdx else (c_vIdx, vIdx)
                eEdgeDict[key] = dist

            if dist > DIST_THRESHOLD[0] and dist_ano > DIST_THRESHOLD[0]:
                key = (c_vIdx, vIdx)
                eEdgeDict[key] = dist

    for k, v in eEdgeDict.items():
        extendEdges.append(k)

    newSegs = []
    for extendE in extendEdges:
        A1, A2 = vs[extendE[1]], vs[extendE[0]] # XY

        minLambda = 1e7
        crossX, crossY = -1, -1
        for seg in segs:
            if seg == extendE or (seg[1], seg[0]) == extendE:
                continue
            B1, B2 = vs[seg[0]], vs[seg[1]]
            insec_stat, X, Y, coeffLambda = calInsecPt(A1, A2, B1, B2)
            if insec_stat and coeffLambda < minLambda:
                minLambda = coeffLambda
                crossX = X
                crossY = Y
            else:
                pass

        if crossX == -1 and crossY == -1:
            continue

        extVec = np.array([crossX - A2[0], crossY - A2[1]])
        minExtDist = np.sqrt(np.dot(extVec, extVec))
        # if minExtDist > DIST_THRESHOLD * EXTEND_UPPER_BOUND * 8 and minLambda > EXTEND_UPPER_BOUND:
        #     crossX = (A2[0] - A1[0]) * EXTEND_UPPER_BOUND + A1[0]
        #     crossY = (A2[1] - A1[1]) * EXTEND_UPPER_BOUND + A1[1]

        if (crossX, crossY) in vs:
            tarIdx = vs.index((crossX, crossY))
        else:
            vs.append((crossX, crossY))
            tarIdx = len(vs) - 1
        edge = (extendE[0], tarIdx) if extendE[0] < tarIdx else (tarIdx, extendE[0])
        newSegs.append(edge)

    segs.extend(newSegs)
    newWalls = dict(vertices=vs, segments=segs)
    return newWalls, newSegs


def calInsecPt(A1, A2, B1, B2):

    Ax, Ay = A1
    Bx, By = A2
    Cx, Cy = B1
    Dx, Dy = B2

    crossAB_CD = (Bx - Ax) * (Dy - Cy) - (By - Ay) * (Dx - Cx)
    if crossAB_CD != 0:
        coeffLambda = ((Cx - Ax) * (Dy - Cy) - (Cy - Ay) * (Dx - Cx)) / crossAB_CD
        coeffMu = ((Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)) / (crossAB_CD * -1)

        if 0 <= coeffMu <= 1 and coeffLambda > 1:
            crossx = Ax + coeffLambda * (Bx - Ax)
            crossy = Ay + coeffLambda * (By - Ay)
            return True, round(crossx) * 1., round(crossy) * 1., coeffLambda
        else:
            return False, 0, 0, coeffLambda

    else:
        if (Cx - Ax) * (Dy - Cy) - (Cy - Ay) * (Dx - Cx) != 0:
            return False, 0, 0, 0
        else:
            A, B, C, D = np.array(A1), np.array(A2), np.array(B1), np.array(B2)
            len_AB = np.linalg.norm(B - A)
            len_AC = np.linalg.norm(A - C)
            len_AD = np.linalg.norm(A - D)
            dotAB_AC = np.dot(B - A, C - A)
            if len_AC == 0:
                cosAB_AC = 0
            else: # 取值正负1
                cosAB_AC = dotAB_AC / (len_AB * len_AC)

            if len_AC < len_AD:
                coeffLambda = len_AC / len_AB * cosAB_AC
                crossx, crossy = B1
            else:
                coeffLambda = len_AD / len_AB * cosAB_AC
                crossx, crossy = B2

            if coeffLambda > 1:
                return True, crossx, crossy, coeffLambda
            else:
                return False, 0, 0, 0



