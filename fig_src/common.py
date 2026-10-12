"""Shared constants, style and helpers for the new_CEP_draft figures.

All figures are drawn from the export in ../../export_z2 (CSV and the TorchScript value
model), so nothing here needs the training repository, acados or a GPU.  The z2 export carries
no set grid, so set slices are evaluated from model/brs_value.pt with the smoothing of the
study's own fig5_headings.py (sigma 0.07 m, outline window 75).
"""
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'export_z2'
OUT = HERE.parent / 'fig'
PREVIEW = HERE / 'preview'     # PNG previews, kept out of fig/ and out of git

# ---------------------------------------------------------------- geometry (geometry.json)
CX, CY = 0.35, 0.175       # pocket half length / half width [m]
TW = 0.15                  # cradle arm thickness [m]
HP = 0.3                   # head point offset [m]
SP = 0.3                   # stern point, astern of the CoG [m]
Y_MS = -0.5                # mothership side boundary [m]
Y_WALL = 0.325             # approach wall, closed for x_head >= -CX [m]
V_S = 0.5                  # stream speed [m/s]
DP = (-1.52, 0.0)          # dynamic positioning point, CoG [m]
TAU = -0.005               # handover threshold
THETA = math.radians(30.0)
U_DOCK = 0.24
XU, UF = -0.1985, 0.0594
F_TRIM = V_S * abs(XU) / UF
DELTA_SCALE = 0.01         # physical rudder angle = 0.01 * delta [rad]
# RaCBF funnel corridor resized for the 0.70 x 0.35 pocket (controller_params.json).
# L is the UPPER bound on y (open side), R the lower bound (mothership side).
FUN = dict(b=3.0, x0=-0.8159, aL=0.1894, y0L=0.3606, aR=0.1641, y0R=0.3359)

CTRL = [('prop', 'Proposed'), ('racbf1', 'RaCBF (1)'), ('racbf2', 'RaCBF (2)')]

# ---------------------------------------------------------------- colours
DOCK, COLL, TOUT = '#2f6ea8', '#c8414b', '#e8912d'      # validated: CVD dE >= 14
PH1, PH2 = '#3b75af', '#c8414b'
SET_EDGE, SET_FILL = '#1b7837', '#a6dba0'
ARM = '#6b6b6b'
TARGET_FILL = '#dcebdc'
MS_FILL, MS_EDGE = '#eadfcb', '#8c6d3f'
WALL_FILL = '#f2f2f2'
FUNNEL = '#7d3c98'
INK, INK2 = '#222222', '#555555'
OUTCOME = {'success': ('dock', DOCK, '*', '-'),
           'crash': ('collision', COLL, 'X', '--'),
           'incomplete': ('timeout', TOUT, 's', '--')}

VCMAP = LinearSegmentedColormap.from_list(
    'v_div', ['#1b7837', '#5aae61', '#a6dba0', '#ffffff', '#fdb863', '#e08214', '#b35806'])


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9,
        'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'axes.linewidth': 0.7, 'axes.edgecolor': '#444444',
        'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'savefig.dpi': 300,
        'pdf.fonttype': 42, 'mathtext.fontset': 'dejavusans',
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(PREVIEW / f'{name}.png', bbox_inches='tight', pad_inches=0.02, dpi=200)
    plt.close(fig)
    print('wrote', OUT / f'{name}.pdf')


# ---------------------------------------------------------------- data
def read_csv(name):
    with open(DATA / name, newline='') as f:
        return list(csv.DictReader(f))


def scenarios():
    """outcomes.csv, with the hand over time and the start position added under the names the
    figure scripts use (prop_t_handover, start_x_rel, start_y_rel)."""
    rows = read_csv('outcomes.csv')
    first = {}
    for r in read_csv('trajectories_prop.csv'):
        s = int(r['scenario'])
        if s not in first:
            first[s] = (float(r['x']) - float(r['CD_x']), float(r['y']))
    for r in rows:
        r['prop_t_handover'] = r.get('prop_t_switch', '')
        r['start_x_rel'], r['start_y_rel'] = (str(v) for v in first[int(r['scenario'])])
    return rows


def trajectories(ctrl):
    """{scenario: dict of float arrays} from trajectories_<ctrl>.csv."""
    out = {}
    for r in read_csv(f'trajectories_{ctrl}.csv'):
        d = out.setdefault(int(r['scenario']), {k: [] for k in r if k != 'scenario'})
        for k, v in r.items():
            if k != 'scenario':
                d[k].append(float(v) if v not in ('', 'nan', 'NaN') else np.nan)
    for d in out.values():
        for k in d:
            d[k] = np.asarray(d[k])
        d['xr'] = d['x'] - d['CD_x']
        d['yr'] = d['y']
        d['xh'] = d['xr'] + HP * np.cos(d['psi'])
        d['yh'] = d['yr'] + HP * np.sin(d['psi'])
    return out


def rel_states(d):
    """(n, 8) cradle relative states in the order the value model expects."""
    return np.stack([d['xr'], d['yr'], d['psi'], d['u'], d['v'], d['r'], d['delta'], d['F']], 1)


_VMODEL = None


def value(states):
    """Learned value V for (n, 8) cradle relative states, from the TorchScript export."""
    global _VMODEL
    import torch
    if _VMODEL is None:
        _VMODEL = torch.jit.load(str(DATA / 'model' / 'brs_value.pt'))
    with torch.no_grad():
        out = [_VMODEL(torch.as_tensor(states[i:i + 200000], dtype=torch.float32)).numpy()
               for i in range(0, len(states), 200000)]
    return np.concatenate(out)


# ---------------------------------------------------------------- set helpers
def avoid_mask(xh, yh, psi, wall=True):
    """True where the bow, CoG or stern violates the collision set (grid in head point frame)."""
    xg, yg = xh, yh
    yc = yg - HP * math.sin(psi)
    ys = yc - SP * math.sin(psi)
    m = (yg < Y_MS) | (yc < Y_MS) | (ys < Y_MS)
    for x0, x1, y0, y1 in arms():
        m |= (xg >= x0) & (xg <= x1) & (yg >= y0) & (yg <= y1)
    if wall:
        m |= (xg >= -CX) & (yg >= Y_WALL)
    return m


def arms():
    return [(CX, CX + TW, -CY - TW, CY + TW),        # back
            (-CX, CX + TW, CY, CY + TW),             # outboard
            (-CX, CX + TW, -CY - TW, -CY)]           # inboard


def mask_from_V(v, av, tg, level=0.0, sigma=0.07, h=0.01, min_island=0.02):
    """{V <= level}, cleaned as in the study's setshape.py: walls filled unsafe, target reached."""
    big = 4.0 * max(abs(level), 0.01)
    vf = np.where(av, +big, np.where(tg, -big, np.nan_to_num(v, nan=+big)))
    m = ndimage.gaussian_filter(vf, sigma / h) <= level
    lab, n = ndimage.label(m)
    if n:
        sz = ndimage.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.where(sz >= min_island / (h * h))[0])
    return ndimage.binary_fill_holes(m) & ~av


def _smooth_poly(p, win):
    if len(p) < win + 3:
        return p
    closed = bool(np.allclose(p[0], p[-1]))
    q = p[:-1] if closed else p
    mode = 'wrap' if closed else 'nearest'
    s = q
    for _ in range(2):
        s = np.stack([ndimage.uniform_filter1d(s[:, 0], win, mode=mode),
                      ndimage.uniform_filter1d(s[:, 1], win, mode=mode)], axis=1)
    return np.vstack([s, s[:1]]) if closed else s


def outline(xs, ys, mask, win=45, min_pts=30):
    fig = plt.figure()
    try:
        cs = fig.gca().contour(xs, ys, mask.astype(float), levels=[0.5])
        segs = cs.allsegs[0]
    finally:
        plt.close(fig)
    return [_smooth_poly(np.asarray(s), win) for s in segs if len(s) >= min_pts]


SET_XL, SET_YL = (-3.0, 0.9), (-0.9, 2.0)


def nominal_state(psi_deg):
    """Nominal slice of the author's fig_set.py: surge u = V_s along the body, v = r = 0, rudder
    centred, trim thrust.  (Earlier versions used zero cradle relative velocity, u = V_s cos psi and
    v = -V_s sin psi, which tilts the set the other way at nonzero heading.)"""
    p = math.radians(psi_deg)
    return (p, V_S, 0.0, 0.0, 0.0, F_TRIM)


def value_slice(state, xlim=SET_XL, ylim=SET_YL, h=0.01):
    """(xs, ys, V, avoid, target) on a head point grid for one (psi, u, v, r, delta, F)."""
    psi, u, v, r, dl, F = state
    xs = np.arange(xlim[0], xlim[1] + 1e-9, h)
    ys = np.arange(ylim[0], ylim[1] + 1e-9, h)
    X, Y = np.meshgrid(xs, ys)
    st = np.zeros((X.size, 8), np.float32)
    st[:, 0] = X.ravel() - HP * math.cos(psi)
    st[:, 1] = Y.ravel() - HP * math.sin(psi)
    st[:, 2:] = [psi, u, v, r, dl, F]
    V = value(st).reshape(X.shape)
    av = avoid_mask(X, Y, psi)
    dxr = u * math.cos(psi) - v * math.sin(psi) - V_S
    dyr = u * math.sin(psi) + v * math.cos(psi)
    ok = abs(psi) <= THETA and abs(dxr) <= U_DOCK and abs(dyr) <= U_DOCK
    tg = (np.abs(X) <= CX) & (np.abs(Y) <= CY) & ok
    return xs, ys, V, av, tg


def wall_shift(xs, ys, av, level, deep, lam):
    """Raise the drawn level toward `deep` near the cradle and its approach wall.

    Returns an array to ADD to the smoothed field so that the outline follows the V = deep contour
    beside the cradle (where the V = level contour runs flat against the wall face) and the
    V = level contour far from it, blended with a Gaussian of width `lam` [m] in the distance to
    the solid cells at x >= -C_X (approach wall and arms; the mothership is left alone).  The shift
    is never negative, so the drawn region only ever shrinks inside the true V <= level set.
    """
    X, Y = np.meshgrid(xs, ys)
    # only the approach wall (the solid water above the cradle), and only from about the top of the
    # cradle upward, so the corridor into the mouth and the pocket itself keep their own outline
    solid = av & (X >= -CX - 1e-9) & (Y >= Y_WALL - 1e-9)
    h = float(xs[1] - xs[0])
    d = ndimage.distance_transform_edt(~solid) * h
    y0, y1 = CY - 0.05, Y_WALL
    ramp = np.clip((Y - y0) / (y1 - y0), 0.0, 1.0)
    ramp = ramp * ramp * (3 - 2 * ramp)
    return (level - deep) * np.exp(-(d / lam) ** 2) * ramp


def corner_weight(xs, ys, xa=-1.5, xb=-0.9):
    """Smooth 0..1 weight for the region beside and above the cradle: x past xa..xb and y above
    the cradle top.  Used to trim the shoulder of the set that rises against the approach wall."""
    X, Y = np.meshgrid(xs, ys)
    sx = np.clip((X - xa) / (xb - xa), 0.0, 1.0)
    sy = np.clip((Y - (CY - 0.05)) / (Y_WALL - (CY - 0.05)), 0.0, 1.0)
    return (sx * sx * (3 - 2 * sx)) * (sy * sy * (3 - 2 * sy))


def author_polys(xs, ys, V, av, level, smooth=4.0, wallfill=0.05, openxy=(110, 6), openell=True,
                 rounding=10, postsmooth=1.0, maskblur=14, shift=None, allow=None):
    """Set outline by the author's pipeline (fig_set.py / SET_FIGURE_SMOOTHING.md, final settings).

    All lengths are in grid cells of a 5 mm grid.  Order: fill the solid cells with `wallfill`,
    Gaussian `smooth`, solid back to 1; morphological opening with a wide flat ellipse
    (`openxy`, semi axes) to cut narrow spikes, closing with a disk of `rounding` to round the
    corners it leaves, kept inside free water; Gaussian `smooth * postsmooth`; blur the set mask by
    `maskblur` and re-cut at a half.  Each panel uses its OWN avoid mask (fig_set.py reads the
    last heading's mask for every panel, which is a slip).  Returns (polygons, processed field).
    """
    v = np.array(V, dtype=float)
    v[av] = wallfill
    if smooth > 0:
        v = ndimage.gaussian_filter(v, smooth)
    v[av] = 1.0
    if shift is not None:
        v = v + shift
    w, h = openxy
    if openell:
        yy, xx = np.ogrid[-h:h + 1, -w:w + 1]
        el = (xx / float(w)) ** 2 + (yy / float(h)) ** 2 <= 1.0
    else:
        el = np.ones((h, w), bool)
    keep = ndimage.binary_opening(v <= level, structure=el)
    if rounding > 0:
        r = int(rounding)
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        keep = ndimage.binary_closing(keep, structure=(xx ** 2 + yy ** 2) <= rounding ** 2)
        keep &= ~av
    if allow is not None:          # optional extra trim; it can only remove cells
        keep &= allow
    v = np.where(keep, np.minimum(v, level - 1e-6), np.maximum(v, level + 1e-6))
    v = ndimage.gaussian_filter(v, max(smooth * postsmooth, 1.0))
    if maskblur > 0:
        m = (ndimage.gaussian_filter((v <= level).astype(float), maskblur) >= 0.5) & ~av
        v = np.where(m, np.minimum(v, level - 1e-6), np.maximum(v, level + 1e-6))
    fig = plt.figure()
    try:
        segs = fig.gca().contour(xs, ys, v, levels=[level]).allsegs[0]
    finally:
        plt.close(fig)
    return [np.asarray(s) for s in segs if len(s) > 20], v


AUTHOR_STEP = 0.005     # the grid the author's smoothing parameters are written for
BAND_DEEP, BAND_OFF = -0.010, 0.15   # candidate Z1 (chosen 2026-10-12): band 0.15 m around V = -0.010


def band_polys(xs, ys, V, av, level, deep=None, off=None):
    """Outline drawn as a band around a deeper contour (candidate Z1).

    The region V <= deep (after the author's wall fill and Gaussian) is grown outward by `off` [m],
    intersected with the author pipeline's V <= level region and smoothed by the same pipeline, so
    the outline runs parallel to the deep contours and never leaves the true V <= level set.
    """
    deep = BAND_DEEP if deep is None else deep
    off = BAND_OFF if off is None else off
    h = float(xs[1] - xs[0])
    vs = np.array(V, dtype=float); vs[av] = 0.05; vs = ndimage.gaussian_filter(vs, 4.0); vs[av] = 1.0
    r = int(round(off / h))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    grown = ndimage.binary_dilation(vs <= deep, structure=(xx ** 2 + yy ** 2) <= r * r)
    return author_polys(xs, ys, V, av, level, allow=grown)[0]


WALL_DEEP, WALL_LAM = -0.018, 0.40   # candidate S3 (chosen 2026-10-12) for the hand over set of Fig. 6


def band_wall_polys(xs, ys, V, av, level):
    """Z1 band plus a deeper level beside the approach wall (candidate S3), so the set narrows into
    the mouth instead of bulging up beside the cradle.  Only removes area."""
    h = float(xs[1] - xs[0])
    vs = np.array(V, dtype=float); vs[av] = 0.05; vs = ndimage.gaussian_filter(vs, 4.0); vs[av] = 1.0
    r = int(round(BAND_OFF / h))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    grown = ndimage.binary_dilation(vs <= BAND_DEEP, structure=(xx ** 2 + yy ** 2) <= r * r)
    return author_polys(xs, ys, V, av, level, allow=grown,
                        shift=wall_shift(xs, ys, av, level, WALL_DEEP, WALL_LAM))[0]


def set_polys(xs, ys, V, av, level, method='author'):
    if method == 'band':
        return band_polys(xs, ys, V, av, level)
    if method == 'band_wall':
        return band_wall_polys(xs, ys, V, av, level)
    return author_polys(xs, ys, V, av, level)[0]


def _nearest_fill(v, av):
    """Solid cells take the value of the nearest free cell, so smoothing does not bleed walls in."""
    idx = ndimage.distance_transform_edt(av, return_distances=False, return_indices=True)
    return v[tuple(idx)]


def wall_feather(xs, ys, V, off=BAND_OFF, rise=0.10):
    """Remove the rise of V within `off` [m] of the approach wall face (author's request 2026-10-12).

    Each row above the wall bottom is capped by its own value `off` left of the face, faded in over
    the second half of that strip and over `rise` [m] above Y_WALL, so the mouth corner keeps its
    own values.  Only lowers V, and only beside the approach wall."""
    v = np.array(V, dtype=float)
    i_far = int(np.argmin(np.abs(xs - (-CX - off))))
    vf = ndimage.gaussian_filter1d(v[:, i_far], 6)
    X, Y = np.meshgrid(xs, ys)
    d = -CX - X
    t = np.clip((d - 0.5 * off) / (0.5 * off), 0.0, 1.0)
    wx = np.where(d < 0, 0.0, 0.5 * (1 + np.cos(np.pi * t)))
    sy = np.clip((Y - Y_WALL) / rise, 0.0, 1.0)
    w = wx * 0.5 * (1 - np.cos(np.pi * sy))
    return v - w * np.maximum(v - vf[:, None], 0.0)


def shade_field(xs, ys, V, av, level, polys, smooth=4.0):
    """Display field for the shaded value map, processed like the outline: the approach wall rise
    feathered, a Gaussian of `smooth` cells, then made consistent with the drawn outline (below
    `level` inside, above outside) and lightly smoothed again."""
    from matplotlib.path import Path as MPath
    v = ndimage.gaussian_filter(_nearest_fill(wall_feather(xs, ys, V), av), smooth)
    X, Y = np.meshgrid(xs, ys)
    pts = np.c_[X.ravel(), Y.ravel()]
    m = np.zeros(X.size, bool)
    for p in polys:
        m ^= MPath(p).contains_points(pts)
    m = m.reshape(X.shape)
    v = np.where(m, np.minimum(v, level - 1e-6), np.maximum(v, level + 1e-6))
    return ndimage.gaussian_filter(_nearest_fill(v, av), 0.5 * smooth)


def boundary(psi_deg, level=0.0, xlim=None, ylim=None, method='author'):
    """Boundary of {V <= level} on the nominal slice at psi_deg (author pipeline, or Z1 band)."""
    xs, ys, V, av, tg = value_slice(nominal_state(psi_deg), xlim or SET_XL, ylim or SET_YL, h=AUTHOR_STEP)
    return set_polys(xs, ys, V, av, level, method)


def slice_set(state, xlim, ylim, h=0.01, level=0.0, sigma=0.07, win=75, method='author'):
    """Set in head point coordinates for one (psi, u, v, r, delta, F), from the value model.

    The outline uses the author's pipeline (or the Z1 band) on a 5 mm grid, like the headings figure.
    """
    xs, ys, V, av, tg = value_slice(state, xlim, ylim, h=AUTHOR_STEP)
    return set_polys(xs, ys, V, av, level, method)


def slice_set_old(state, xlim, ylim, h=0.01, level=0.0, sigma=0.07, win=75):
    """Previous outline (blurred mask contour with a moving average), kept for comparison."""
    psi, u, v, r, dl, F = state
    xs = np.arange(xlim[0], xlim[1] + 1e-9, h)
    ys = np.arange(ylim[0], ylim[1] + 1e-9, h)
    X, Y = np.meshgrid(xs, ys)
    n = X.size
    st = np.zeros((n, 8), np.float32)
    st[:, 0] = X.ravel() - HP * math.cos(psi)
    st[:, 1] = Y.ravel() - HP * math.sin(psi)
    st[:, 2:] = [psi, u, v, r, dl, F]
    V = value(st).reshape(X.shape)
    av = avoid_mask(X, Y, psi)
    dxr = u * math.cos(psi) - v * math.sin(psi) - V_S
    dyr = u * math.sin(psi) + v * math.cos(psi)
    ok = abs(psi) <= THETA and abs(dxr) <= U_DOCK and abs(dyr) <= U_DOCK
    tg = (np.abs(X) <= CX) & (np.abs(Y) <= CY) & ok
    return outline(xs, ys, mask_from_V(V, av, tg, level, sigma=sigma, h=h), win=win)


# ---------------------------------------------------------------- drawing
def draw_geometry(ax, xlim, ylim, wall=False, target=True):
    ax.axhspan(ylim[0] - 5, Y_MS, color=MS_FILL, zorder=1, lw=0)
    ax.axhline(Y_MS, color=MS_EDGE, lw=0.9, zorder=2)
    if wall:
        ax.add_patch(Rectangle((-CX, Y_WALL), 10, 10, fc='none', ec='#9a9a9a', lw=0,
                               hatch='////', zorder=1.5))
    if target:
        ax.add_patch(Rectangle((-CX, -CY), 2 * CX, 2 * CY, fc=TARGET_FILL, ec='none', zorder=2))
    for x0, x1, y0, y1 in arms():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=ARM, ec='none', zorder=6))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')


SET_LINE = '#333333'        # outline colour of the headings figure, reused wherever a set is drawn


def draw_set(ax, polys, fill=True, ls='-', lw=0.9, alpha=0.45, z=3, label=None, edge=SET_LINE):
    for i, p in enumerate(polys):
        if fill:
            ax.add_patch(Polygon(p, closed=True, fc=SET_FILL, ec='none', alpha=alpha, zorder=z))
        ax.plot(p[:, 0], p[:, 1], color=edge, lw=lw, ls=ls, zorder=z + 0.1,
                label=label if i == 0 else None)


def hull(ax, cx, cy, psi, fc, ec, alpha=0.55, z=8, lw=0.7):
    """USV outline at model scale (0.60 x 0.25 m), bow at the head point, stern SP astern of the CoG."""
    hp, beam, aft = HP, 0.125, SP
    pts = np.array([[hp, 0.0], [hp - 0.05, 0.07], [hp - 0.16, beam], [-aft, beam * 0.85],
                    [-aft, -beam * 0.85], [hp - 0.16, -beam], [hp - 0.05, -0.07]])
    c, s = math.cos(psi), math.sin(psi)
    pts = pts @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z))


def funnel_curves(x):
    """(upper, lower) funnel bounds on the head point y.  L sets the upper (open) side, R the lower."""
    t = np.tanh(FUN['b'] * (FUN['x0'] - x))
    return FUN['aL'] * t + FUN['y0L'], -(FUN['aR'] * t + FUN['y0R'])
