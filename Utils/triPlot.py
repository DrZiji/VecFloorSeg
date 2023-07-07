import numpy as np


lw = 0.5
scatterS = 0.5

def compare(plt, A, B, figsize=(6, 3), plot_label=True):
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(231)
    plot(ax1, **A)
    # lim = ax1.axis()
    ax2 = plt.subplot(232)
    plot(ax2, **B)
    # ax2.axis(lim)
    if plot_label:
        ax3 = plt.subplot(234)
        plotLabel(ax3, **B)
    # ax3.axis(lim)
    plt.tight_layout()


def comparev(plt, A, B, figsize=(3, 6)):
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(221)
    plot(ax1, **A)
    lim = ax1.axis()
    ax2 = plt.subplot(222, sharex=ax1)
    plot(ax2, **B)
    ax3 = plt.subplot(223, sharex=ax1)
    plotLabel(ax3,  **A)
    # ax2.axis(lim)
    plt.tight_layout()


def plot(ax, **kw):
    ax.set_aspect('equal')
    vertices(ax, **kw)
    if 'segments' in kw:
        segments(ax, **kw)
    if 'triangles' in kw:
        triangles(ax, **kw)
    if 'holes' in kw:
        holes(ax, **kw)
    if 'edges' in kw:
        edges(ax, **kw)
    if 'regions' in kw:
        regions(ax, **kw)
    if 'triangle_attributes' in kw:
        triangle_attributes(ax, **kw)

    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()  # 反转Y坐标轴
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def plotLabel(ax, plotCoeff='triangles_label', **kw):
    ax.axes.set_aspect('equal')
    verts = np.array(kw['vertices'])
    assert plotCoeff in ['triangles_label', 'preds', 'comparison'], 'the wrong specification.'
    if plotCoeff in kw:
        if plotCoeff == 'comparison':
            ax.tripcolor(
                verts[:, 0], verts[:, 1], triangles=kw['triangles'],
                facecolors=kw[plotCoeff].squeeze(axis=1), vmin=0, vmax=15
            )
        else:
            ax.tripcolor(
                verts[:, 0], verts[:, 1], triangles=kw['triangles'],
                facecolors=kw[plotCoeff].squeeze(axis=1), vmin=0, vmax=255, cmap=kw['cmap']
            )
        #
    else:
        raise NotImplementedError

    ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    ax.invert_yaxis()  # 反转Y坐标轴
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def vertices(ax, **kw):
    verts = np.array(kw['vertices'])
    ax.scatter(*verts.T, color='k', s=scatterS)
    if 'labels' in kw:
        for i in range(verts.shape[0]):
            ax.text(verts[i, 0], verts[i, 1], str(i))
    if 'markers' in kw:
        vm = kw['vertex_markers']
        for i in range(verts.shape[0]):
            ax.text(verts[i, 0], verts[i, 1], str(vm[i]))


def segments(ax, color='black', **kw):
    verts = np.array(kw['vertices'])
    segs = np.array(kw['segments'])
    for beg, end in segs:
        x0, y0 = verts[beg, :]
        x1, y1 = verts[end, :]
        ax.fill(
            [x0, x1],
            [y0, y1],
            facecolor='none',
            edgecolor=color,
            linewidth=lw,
            zorder=0,
        )


def triangles(ax, **kw):
    verts = np.array(kw['vertices'])
    ax.triplot(verts[:, 0], verts[:, 1], kw['triangles'], 'ko-', ms=scatterS)


def holes(ax, **kw):
    holes = np.array(kw['holes'])
    ax.scatter(*holes.T, marker='x', color='r', s=scatterS)


def edges(ax, **kw):
    """
    Plot regular edges and rays (edges whose one endpoint is at infinity)
    """
    verts = kw['vertices']
    edges = kw['edges']
    for beg, end in edges:
        x0, y0 = verts[beg, :]
        x1, y1 = verts[end, :]
        ax.fill(
            [x0, x1],
            [y0, y1],
            facecolor='none',
            edgecolor='k',
            linewidth=.5,
        )

    if ('ray_origins' not in kw) or ('ray_directions' not in kw):
        return

    lim = ax.axis()
    ray_origin = kw['ray_origins']
    ray_direct = kw['ray_directions']
    for (beg, (vx, vy)) in zip(ray_origin.flatten(), ray_direct):
        x0, y0 = verts[beg, :]
        scale = 100.0  # some large number
        x1, y1 = x0 + scale * vx, y0 + scale * vy
        ax.fill(
            [x0, x1],
            [y0, y1],
            facecolor='none',
            edgecolor='k',
            linewidth=.5,
        )
    ax.axis(lim)  # make sure figure is not rescaled by ifinite ray


def regions(ax, **kw):
    """
    Plot regions labeled by region
    """
    regions = np.array(kw['regions'])
    ax.scatter(regions[:, 0], regions[:, 1], marker='*', color='b', s=scatterS)
    for x, y, r, _ in regions:
        ax.text(x, y, ' {:g}'.format(r), color='b', va='center')


def triangle_attributes(ax, **kw):
    """
    Plot triangle attributes labeled by region
    """
    verts = np.array(kw['vertices'])
    tris = np.array(kw['triangles'])
    attrs = np.array(kw['triangle_attributes']).flatten()
    centroids = verts[tris].mean(axis=1)
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker='.',
        color='m',
        zorder=1,
        s=scatterS
    )
    for (x, y), r in zip(centroids, attrs):
        ax.text(x, y, ' {:g}'.format(r), color='m', zorder=1, va='center')
