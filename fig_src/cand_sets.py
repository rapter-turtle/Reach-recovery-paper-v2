"""Candidate looks for the learned set figure (Fig. 5), for the author to pick the cleanest shape.

    python cand_sets.py   -> preview/candidates/sets/set_<A..H>.png and set_gallery.png

Every row uses the manuscript design (value shading, solid outline, cradle, mothership, approach
wall) and varies only how the set is sliced and outlined:
  slice  'author' = surge V_s along the body, v = r = 0 (fig_set.py)
         'zerorel' = zero cradle relative velocity, u = V_s cos psi, v = -V_s sin psi
  level  the value level drawn as the boundary
  method 'final' = fig_set.py final settings, 'nomaskblur' = the same without the mask blur,
         'previous' = blurred mask contour with a moving average (the earlier manuscript outline)
"""
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import ff_common as F
import common as C

C.style()
OUT = F.OUTDIR / 'sets'
OUT.mkdir(parents=True, exist_ok=True)
XL, YL = C.SET_XL, C.SET_YL
H = C.AUTHOR_STEP

CANDS = [   # key, slice, headings, level, method, note
    ('A', 'author', 10, -0.006, 'final', "author's figure: u = 0.5, v = 0, +-10 deg, V = -0.006"),
    ('B', 'author', 15, -0.006, 'final', 'author slice, +-15 deg'),
    ('C', 'zerorel', 10, -0.006, 'final', 'zero relative velocity slice, +-10 deg'),
    ('D', 'zerorel', 15, -0.006, 'final', 'zero relative velocity slice, +-15 deg (last local version)'),
    ('E', 'author', 10, -0.005, 'final', 'author slice, +-10 deg, boundary at tau = -0.005 (the run)'),
    ('F', 'author', 10, 0.0, 'final', 'author slice, +-10 deg, boundary at V = 0 (whole set)'),
    ('G', 'author', 10, -0.006, 'nomaskblur', 'author slice, +-10 deg, final settings without the mask blur'),
    ('H', 'author', 10, -0.006, 'previous', 'author slice, +-10 deg, previous outline method'),
]
CANDS15 = [  # the same comparison at +-15 deg
    ('I', 'author', 15, -0.006, 'final', 'author slice, +-15 deg, V = -0.006 (same as B)'),
    ('J', 'zerorel', 15, -0.006, 'final', 'zero relative velocity slice, +-15 deg, V = -0.006 (same as D)'),
    ('K', 'author', 15, -0.005, 'final', 'author slice, +-15 deg, boundary at tau = -0.005 (the run)'),
    ('L', 'zerorel', 15, -0.005, 'final', 'zero relative velocity slice, +-15 deg, tau = -0.005'),
    ('M', 'author', 15, 0.0, 'final', 'author slice, +-15 deg, boundary at V = 0 (whole set)'),
    ('N', 'zerorel', 15, 0.0, 'final', 'zero relative velocity slice, +-15 deg, V = 0'),
    ('O', 'zerorel', 15, -0.006, 'nomaskblur', 'zero relative velocity slice, +-15 deg, without the mask blur'),
    ('P', 'zerorel', 15, -0.006, 'previous', 'zero relative velocity slice, +-15 deg, previous outline method'),
]
_cache = {}


def state(kind, deg):
    p = math.radians(deg)
    if kind == 'author':
        return (p, C.V_S, 0.0, 0.0, 0.0, C.F_TRIM)
    return (p, C.V_S * math.cos(p), -C.V_S * math.sin(p), 0.0, 0.0, C.F_TRIM)


def field(kind, deg):
    key = (kind if deg else 'author', deg)          # psi = 0 is the same slice either way
    if key not in _cache:
        _cache[key] = C.value_slice(state(kind, deg), XL, YL, h=H)
    return _cache[key]


def outline(xs, ys, V, av, tg, level, method):
    if method == 'final':
        return C.author_polys(xs, ys, V, av, level)[0]
    if method == 'nomaskblur':
        return C.author_polys(xs, ys, V, av, level, maskblur=0)[0]
    return C.outline(xs, ys, C.mask_from_V(V, av, tg, level, sigma=0.07, h=H), win=150)


def draw_row(axs, kind, psi, level, method):
    norm = TwoSlopeNorm(vmin=-0.04, vcenter=0.0, vmax=0.04)
    cmap = C.VCMAP.copy(); cmap.set_bad('#e9e9e9')
    for ax, deg in zip(axs, (-psi, 0, psi)):
        xs, ys, V, av, tg = field(kind, deg)
        im = ax.pcolormesh(xs, ys, np.where(av, np.nan, np.clip(V, -0.04, 0.04)), cmap=cmap, norm=norm,
                           shading='auto', rasterized=True, zorder=0.5)
        C.draw_geometry(ax, XL, YL, wall=True, target=False)
        ax.add_patch(plt.Rectangle((-C.CX, -C.CY), 2 * C.CX, 2 * C.CY, fc='none', ec=C.SET_EDGE, lw=0.6,
                                   ls=':', zorder=5))
        for p in outline(xs, ys, V, av, tg, level, method):
            ax.plot(p[:, 0], p[:, 1], color=C.SET_LINE, lw=0.9, zorder=4)
        ax.plot(C.DP[0] + C.HP, C.DP[1], 'D', ms=3.2, color='#333333', mec='white', mew=0.4, zorder=10)
        ax.set_title(r'$\psi_r = %+d^\circ$' % deg if deg else r'$\psi_r = 0^\circ$', fontsize=8.5)
        ax.set_xticks([-3, -2, -1, 0])
    return im


def legend(fig, y):
    fig.legend(handles=[Patch(fc=C.ARM, label='cradle'), Patch(fc=C.MS_FILL, ec=C.MS_EDGE, label='mothership'),
                        Patch(fc='#e9e9e9', ec='#9a9a9a', hatch='////', lw=0, label='approach wall'),
                        Line2D([], [], color=C.SET_LINE, lw=0.9, label='set boundary'),
                        Line2D([], [], ls='', marker='D', ms=3.2, color='#333333', label='DP point')],
               loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, y), fontsize=7.5)


import sys
if '--psi15' in sys.argv:
    CANDS, GALLERY = CANDS15, 'set_gallery_15deg.png'
else:
    GALLERY = 'set_gallery.png'

# one file per candidate
for key, kind, psi, level, method, note in CANDS:
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True, gridspec_kw=dict(wspace=0.06))
    im = draw_row(axs, kind, psi, level, method)
    for ax in axs:
        ax.set_xlabel(r'$x_h$ [m]')
    axs[0].set_ylabel(r'$y_h$ [m]')
    cb = fig.colorbar(im, cax=fig.add_axes([0.915, 0.2, 0.012, 0.62]))
    cb.set_label('value $V$', fontsize=8); cb.ax.tick_params(labelsize=7)
    legend(fig, -0.12)
    fig.suptitle(f'{key}   {note}', fontsize=8.5, y=1.03, color=C.INK2)
    fig.savefig(OUT / f'set_{key}.png', dpi=200, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print('wrote', OUT / f'set_{key}.png')

# gallery: all candidates stacked
n = len(CANDS)
fig, axs = plt.subplots(n, 3, figsize=(7.2, 1.95 * n), sharex=True, sharey=True,
                        gridspec_kw=dict(wspace=0.06, hspace=0.42))
for r, (key, kind, psi, level, method, note) in enumerate(CANDS):
    draw_row(axs[r], kind, psi, level, method)
    axs[r][0].set_ylabel(r'$y_h$ [m]', fontsize=8)
    axs[r][0].annotate(key, xy=(-0.32, 0.5), xycoords='axes fraction', fontsize=15, fontweight='bold',
                       ha='center', va='center', color='#1f2d4a')
    axs[r][1].text(0.5, 1.2, note, transform=axs[r][1].transAxes, ha='center', va='bottom', fontsize=7.5,
                   color=C.INK2)
for ax in axs[-1]:
    ax.set_xlabel(r'$x_h$ [m]')
legend(fig, 0.065)
fig.savefig(OUT / GALLERY, dpi=150, bbox_inches='tight', pad_inches=0.05)
print('wrote', OUT / GALLERY)
