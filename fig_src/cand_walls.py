"""Candidate A (author slice, +-10 deg, V = -0.006, author pipeline) with the outline pulled off the
cradle wall so it runs parallel to the deeper contours there.

    python cand_walls.py  -> preview/candidates/sets/wall_gallery.png (+ wall_<key>.png)

Near the approach wall (above the cradle top) the drawn level slides from V = -0.006 toward a deeper level
(common.wall_shift); far from them it stays -0.006.  The thin blue lines are the smoothed V = -0.008,
-0.010 and -0.012 contours, for judging how parallel the outline runs to them.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy import ndimage

import types

import ff_common as F  # finalfinal geometry and model
import common as C

C.style()
_cache = {}


def _field(kind, deg):
    if deg not in _cache:
        _cache[deg] = C.value_slice(C.nominal_state(deg), C.SET_XL, C.SET_YL, h=C.AUTHOR_STEP)
    return _cache[deg]


S = types.SimpleNamespace(field=_field, XL=C.SET_XL, YL=C.SET_YL, OUT=F.OUTDIR / 'sets')
S.OUT.mkdir(parents=True, exist_ok=True)
LEVEL = -0.006
CANDS = [   # key, deep level, blend width [m], wallfill, note
    ('A', None, None, 0.05, 'A as before'),
    ('W1', -0.008, 0.25, 0.05, 'beside the wall follow V = -0.008, blend 0.25 m'),
    ('W2', -0.009, 0.30, 0.05, 'beside the wall follow V = -0.009, blend 0.30 m'),
    ('W3', -0.010, 0.35, 0.05, 'beside the wall follow V = -0.010, blend 0.35 m'),
    ('W4', -0.009, 0.45, 0.05, 'beside the wall follow V = -0.009, wider blend 0.45 m'),
    ('W5', -0.012, 0.40, 0.05, 'beside the wall follow V = -0.012, blend 0.40 m'),
    ('W6', None, None, 0.30, 'no level shift, wall fill raised to 0.30 (fig_set.py knob)'),
]


def row(axs, deep, lam, wallfill):
    norm = TwoSlopeNorm(vmin=-0.04, vcenter=0.0, vmax=0.04)
    cmap = C.VCMAP.copy(); cmap.set_bad('#e9e9e9')
    for ax, deg in zip(axs, (-10, 0, 10)):
        xs, ys, V, av, tg = S.field('author', deg)
        ax.pcolormesh(xs, ys, np.where(av, np.nan, np.clip(V, -0.04, 0.04)), cmap=cmap, norm=norm,
                      shading='auto', rasterized=True, zorder=0.5)
        C.draw_geometry(ax, S.XL, S.YL, wall=True, target=False)
        ax.add_patch(plt.Rectangle((-C.CX, -C.CY), 2 * C.CX, 2 * C.CY, fc='none', ec=C.SET_EDGE, lw=0.6,
                                   ls=':', zorder=5))
        vs = V.copy(); vs[av] = 0.05; vs = ndimage.gaussian_filter(vs, 4.0); vs[av] = 1.0
        ax.contour(xs, ys, vs, levels=[-0.012, -0.010, -0.008], colors=['#2171b5'], linewidths=0.45,
                   alpha=0.8, zorder=3.5)
        shift = None if deep is None else C.wall_shift(xs, ys, av, LEVEL, deep, lam)
        for p in C.author_polys(xs, ys, V, av, LEVEL, wallfill=wallfill, shift=shift)[0]:
            ax.plot(p[:, 0], p[:, 1], color=C.SET_LINE, lw=1.0, zorder=4)
        ax.plot(C.DP[0] + C.HP, C.DP[1], 'D', ms=3.2, color='#333333', mec='white', mew=0.4, zorder=10)
        ax.set_title(r'$\psi_r = %+d^\circ$' % deg if deg else r'$\psi_r = 0^\circ$', fontsize=8.5)
        ax.set_xticks([-3, -2, -1, 0])


n = len(CANDS)
fig, axs = plt.subplots(n, 3, figsize=(7.2, 1.95 * n), sharex=True, sharey=True,
                        gridspec_kw=dict(wspace=0.06, hspace=0.42))
for r, (key, deep, lam, wf, note) in enumerate(CANDS):
    row(axs[r], deep, lam, wf)
    axs[r][0].set_ylabel(r'$y_h$ [m]', fontsize=8)
    axs[r][0].annotate(key, xy=(-0.34, 0.5), xycoords='axes fraction', fontsize=14, fontweight='bold',
                       ha='center', va='center', color='#1f2d4a')
    axs[r][1].text(0.5, 1.2, note, transform=axs[r][1].transAxes, ha='center', va='bottom', fontsize=7.5,
                   color=C.INK2)
    f1, a1 = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True, gridspec_kw=dict(wspace=0.06))
    row(a1, deep, lam, wf)
    f1.suptitle(f'{key}   {note}', fontsize=8.5, y=1.03, color=C.INK2)
    f1.savefig(S.OUT / f'wall_{key}.png', dpi=200, bbox_inches='tight', pad_inches=0.03)
    plt.close(f1)
for ax in axs[-1]:
    ax.set_xlabel(r'$x_h$ [m]')
fig.legend(handles=[Line2D([], [], color=C.SET_LINE, lw=1.0, label='drawn outline'),
                    Line2D([], [], color='#2171b5', lw=0.6, label=r'smoothed $V$ = $-0.008$, $-0.010$, $-0.012$'),
                    Line2D([], [], ls='', marker='D', ms=3.2, color='#333333', label='DP point')],
           loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.045), fontsize=7.5)
fig.savefig(S.OUT / 'wall_gallery.png', dpi=150, bbox_inches='tight', pad_inches=0.05)
print('wrote', S.OUT / 'wall_gallery.png')
