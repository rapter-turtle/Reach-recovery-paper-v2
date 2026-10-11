"""Candidate A with the shoulder beside the approach wall removed (the region the author circled).

    python cand_corner.py -> preview/candidates/sets/corner_gallery.png (+ corner_<key>.png)

In the region beside and above the cradle (common.corner_weight) the outline follows a deeper
contour, either the contour itself (level shift) or a copy of it moved outward by a fixed distance
(parallel offset).  Elsewhere it stays the V = -0.006 outline of candidate A.  Both only remove area.
"""
import types

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy import ndimage

import ff_common as F
import common as C

C.style()
LEVEL = -0.006
OUT = F.OUTDIR / 'sets'
OUT.mkdir(parents=True, exist_ok=True)
_cache = {}
CANDS = [   # key, mode, deep level, offset [m], note
    ('A', None, None, None, 'A as before'),
    ('X1', 'shift', -0.010, None, 'beside the cradle follow V = -0.010'),
    ('X2', 'shift', -0.012, None, 'beside the cradle follow V = -0.012'),
    ('X3', 'offset', -0.012, 0.15, 'beside the cradle: V = -0.012 moved out by 0.15 m'),
    ('X4', 'offset', -0.012, 0.25, 'beside the cradle: V = -0.012 moved out by 0.25 m'),
]


def field(deg):
    if deg not in _cache:
        _cache[deg] = C.value_slice(C.nominal_state(deg), C.SET_XL, C.SET_YL, h=C.AUTHOR_STEP)
    return _cache[deg]


def polys(xs, ys, V, av, mode, deep, off):
    if mode is None:
        return C.author_polys(xs, ys, V, av, LEVEL)[0]
    w = C.corner_weight(xs, ys)
    if mode == 'shift':
        return C.author_polys(xs, ys, V, av, LEVEL, shift=(LEVEL - deep) * w)[0]
    vs = V.copy(); vs[av] = 0.05; vs = ndimage.gaussian_filter(vs, 4.0); vs[av] = 1.0
    r = int(round(off / C.AUTHOR_STEP))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    grown = ndimage.binary_dilation(vs <= deep, structure=(xx ** 2 + yy ** 2) <= r * r)
    return C.author_polys(xs, ys, V, av, LEVEL, allow=grown | (w < 0.5))[0]


def row(axs, mode, deep, off):
    norm = TwoSlopeNorm(vmin=-0.04, vcenter=0.0, vmax=0.04)
    cmap = C.VCMAP.copy(); cmap.set_bad('#e9e9e9')
    for ax, deg in zip(axs, (-10, 0, 10)):
        xs, ys, V, av, tg = field(deg)
        ax.pcolormesh(xs, ys, np.where(av, np.nan, np.clip(V, -0.04, 0.04)), cmap=cmap, norm=norm,
                      shading='auto', rasterized=True, zorder=0.5)
        C.draw_geometry(ax, C.SET_XL, C.SET_YL, wall=True, target=False)
        ax.add_patch(plt.Rectangle((-C.CX, -C.CY), 2 * C.CX, 2 * C.CY, fc='none', ec=C.SET_EDGE, lw=0.6,
                                   ls=':', zorder=5))
        vs = V.copy(); vs[av] = 0.05; vs = ndimage.gaussian_filter(vs, 4.0); vs[av] = 1.0
        ax.contour(xs, ys, vs, levels=[-0.012, -0.010, -0.008], colors=['#2171b5'], linewidths=0.45,
                   alpha=0.8, zorder=3.5)
        for p in polys(xs, ys, V, av, mode, deep, off):
            ax.plot(p[:, 0], p[:, 1], color=C.SET_LINE, lw=1.0, zorder=4)
        ax.plot(C.DP[0] + C.HP, C.DP[1], 'D', ms=3.2, color='#333333', mec='white', mew=0.4, zorder=10)
        ax.set_title(r'$\psi_r = %+d^\circ$' % deg if deg else r'$\psi_r = 0^\circ$', fontsize=8.5)
        ax.set_xticks([-3, -2, -1, 0])


n = len(CANDS)
fig, axs = plt.subplots(n, 3, figsize=(7.2, 1.95 * n), sharex=True, sharey=True,
                        gridspec_kw=dict(wspace=0.06, hspace=0.42))
for r_, (key, mode, deep, off, note) in enumerate(CANDS):
    row(axs[r_], mode, deep, off)
    axs[r_][0].set_ylabel(r'$y_h$ [m]', fontsize=8)
    axs[r_][0].annotate(key, xy=(-0.34, 0.5), xycoords='axes fraction', fontsize=14, fontweight='bold',
                        ha='center', va='center', color='#1f2d4a')
    axs[r_][1].text(0.5, 1.2, note, transform=axs[r_][1].transAxes, ha='center', va='bottom', fontsize=7.5,
                    color=C.INK2)
    f1, a1 = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True, gridspec_kw=dict(wspace=0.06))
    row(a1, mode, deep, off)
    f1.suptitle(f'{key}   {note}', fontsize=8.5, y=1.03, color=C.INK2)
    f1.savefig(OUT / f'corner_{key}.png', dpi=200, bbox_inches='tight', pad_inches=0.03)
    plt.close(f1)
for ax in axs[-1]:
    ax.set_xlabel(r'$x_h$ [m]')
fig.legend(handles=[Line2D([], [], color=C.SET_LINE, lw=1.0, label='drawn outline'),
                    Line2D([], [], color='#2171b5', lw=0.6, label=r'smoothed $V$ = $-0.008$, $-0.010$, $-0.012$'),
                    Line2D([], [], ls='', marker='D', ms=3.2, color='#333333', label='DP point')],
           loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.04), fontsize=7.5)
fig.savefig(OUT / 'corner_gallery.png', dpi=150, bbox_inches='tight', pad_inches=0.05)
print('wrote', OUT / 'corner_gallery.png')
