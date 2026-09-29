"""Exact CPU speed-ups for the BT-1 inference path (BT-3, M3-08). Outputs are bit-identical to the reference path;
tests/test_fastops.py checks that on dev5 part maps. Nothing here changes a threshold, a model or a feature.

  1. binary_dilation with the default cross structure and iterations k: OpenCV dilate with the same cross kernel.
     scipy pads with 0 outside the image; OpenCV's default border value for dilate never wins a max, so the two agree.
  2. geodesic_fast: the pixel graph is built on the bounding box of the passable pixels (paths never leave them), and
     cached for the three distance calls that TA.analyse makes on the same limb.

apply() patches the modules the engine uses (test_a_masks, test_e2, side_kp, lend, eval_v3, engine_bt1). Training code
is untouched unless it calls apply().
"""
import numpy as np, cv2
from scipy import ndimage as _nd, sparse
from scipy.sparse.csgraph import dijkstra

_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


def binary_dilation(a, structure=None, iterations=1, **kw):
    if structure is not None or kw or iterations < 1 or np.ndim(a) != 2:
        return _nd.binary_dilation(a, structure=structure, iterations=iterations, **kw)
    m = np.asarray(a, bool)
    if not m.any():
        return m.copy()
    ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    H, W = m.shape; k = int(iterations)
    y0, y1, x0, x1 = max(ys[0] - k, 0), min(ys[-1] + k + 1, H), max(xs[0] - k, 0), min(xs[-1] + k + 1, W)
    out = np.zeros_like(m)
    out[y0:y1, x0:x1] = cv2.dilate(m[y0:y1, x0:x1].view(np.uint8), _CROSS, iterations=k).astype(bool)
    return out


class _ND:
    """scipy.ndimage stand-in: fast binary_dilation, everything else passed through."""
    binary_dilation = staticmethod(binary_dilation)

    def __getattr__(self, name):
        return getattr(_nd, name)


ND = _ND()
_cache = {}


def geodesic_fast(limb, passable, seeds):
    ok = limb | passable
    dist = np.full(limb.shape, np.inf)
    ry, rx = np.nonzero(ok.any(1))[0], np.nonzero(ok.any(0))[0]
    if len(ry) == 0:
        return dist
    y0, y1, x0, x1 = ry[0], ry[-1] + 1, rx[0], rx[-1] + 1
    okc = ok[y0:y1, x0:x1]
    key = (y0, x0, okc.shape, np.packbits(okc).tobytes())
    g = _cache.get('g')
    if g is None or g[0] != key:
        ys, xs = np.nonzero(okc); n = len(ys); H, W = okc.shape
        idx = -np.ones(okc.shape, np.int64); idx[ys, xs] = np.arange(n)
        r, c, w = [], [], []
        for dy, dx, cost in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.414), (1, -1, 1.414)):
            y2, x2 = ys + dy, xs + dx
            v = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
            v[v] = okc[y2[v], x2[v]]
            r.append(idx[ys[v], xs[v]]); c.append(idx[y2[v], x2[v]]); w.append(np.full(v.sum(), cost))
        G = sparse.coo_matrix((np.concatenate(w), (np.concatenate(r), np.concatenate(c))), shape=(n, n)).tocsr()
        g = (key, ys, xs, idx, G); _cache['g'] = g
    _, ys, xs, idx, G = g
    src = idx[(seeds & ok)[y0:y1, x0:x1]]
    if len(src) == 0:
        return dist
    d = dijkstra(G, directed=False, indices=np.unique(src), min_only=True)
    dc = np.full(okc.shape, np.inf); dc[ys, xs] = d
    dist[y0:y1, x0:x1] = dc
    dist[~limb] = np.inf
    return dist


def apply():
    import test_a_masks as TA, test_e2 as T2, side_kp as SK, lend as LE, eval_v3 as EV
    for mod in (TA, T2, SK, LE, EV):
        mod.ndimage = ND
    TA.geodesic = geodesic_fast
    SK.geodesic_fast = geodesic_fast
    try:
        import engine_bt1 as EB
        EB.ndimage = ND
    except ImportError:
        pass
