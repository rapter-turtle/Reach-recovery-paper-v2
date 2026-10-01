"""Learned robust reach avoid set at three headings (psi_r = -15, 0, +15 deg).

Nominal slice: zero cradle relative velocity (u = V_s), v = r = 0, rudder centred, trim thrust,
drawn in the head point frame (x_h, y_h) the reach and avoid conditions are read at.
The value field comes from set_value_grid.npz and the boundaries from set_boundary.csv.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common as C

C.style()
g = np.load(C.DATA / 'set_value_grid.npz')
xs, ys = g['x_head'], g['y_head']
XL, YL = (-3.2, 1.0), (-0.9, 1.6)
psis = [('-15', -15), ('+00', 0), ('+15', 15)]

vmin, vmax = -0.04, 0.06
norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
cmap = C.VCMAP.copy()
cmap.set_bad('#e9e9e9')

fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True,
                        gridspec_kw=dict(wspace=0.06))
for ax, (key, deg) in zip(axs, psis):
    V = np.array(g[f'V_psi{key}'], dtype=float)
    av = g[f'avoid_psi{key}']
    V = np.where(av, np.nan, np.clip(V, vmin, vmax))
    im = ax.pcolormesh(xs, ys, V, cmap=cmap, norm=norm, shading='auto', rasterized=True, zorder=0.5)
    C.draw_geometry(ax, XL, YL, wall=True, target=False)
    ax.add_patch(plt.Rectangle((-C.CX, -C.CY), 2 * C.CX, 2 * C.CY, fc='none', ec=C.SET_EDGE,
                               lw=0.6, ls=':', zorder=5))
    for p in C.boundary(deg, 0.0):
        ax.plot(p[:, 0], p[:, 1], color='#333333', lw=0.9, zorder=4)
    ax.set_title(r'$\psi_r = %+d^\circ$' % deg if deg else r'$\psi_r = 0^\circ$')
    ax.set_xlabel(r'$x_h$ [m]')
    ax.set_xticks([-3, -2, -1, 0, 1])
axs[0].set_ylabel(r'$y_h$ [m]')

cax = fig.add_axes([0.915, 0.2, 0.012, 0.62])
cb = fig.colorbar(im, cax=cax)
cb.set_label('value $V$', fontsize=8.5)
cb.set_ticks([-0.04, -0.02, 0.0, 0.02, 0.04, 0.06])
cb.ax.tick_params(labelsize=7)

handles = [Patch(fc=C.ARM, ec='none', label='cradle'),
           Patch(fc=C.MS_FILL, ec=C.MS_EDGE, lw=0.8, label='mothership'),
           Patch(fc='#e9e9e9', ec='#9a9a9a', hatch='////', lw=0, label='approach wall'),
           Line2D([], [], color='#333333', lw=0.9, label=r'set boundary $V=0$')]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.1), handlelength=1.6, columnspacing=1.2)
C.save(fig, 'reach_avoid_set')
