"""Head point paths of five representative scenarios under each controller (export_z2).

The five are fixed so that every baseline failure mode appears:
      85   RaCBF (1) stays outside the funnel corridor until the mouth, then strikes the back
           of the cradle (the only RaCBF (1) collision that leaves the corridor on the final
           approach; the other three stay inside it and hit the back wall on speed)
   12, 72  RaCBF (2) stalls astern of the mouth, RaCBF (1) docks
    0, 28  all three controllers dock
The proposed panel shades the learned set at the DP point (nominal slice, psi_r = 0),
the baseline panels shade the funnel corridor the RaCBF enforces in its docking phase.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common as C

C.style()
FIVE = [85, 12, 72, 0, 28]
S = {int(r['scenario']): r for r in C.scenarios()}
XL, YL = (-4.6, 1.0), (-0.75, 2.5)

fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.05), sharey=True, gridspec_kw=dict(wspace=0.05))
for ax, (k, lab) in zip(axs, C.CTRL):
    C.draw_geometry(ax, XL, YL, wall=False, target=True)
    if k == 'prop':
        C.draw_set(ax, C.boundary(0, C.TAU), alpha=0.4, z=2.5)
    else:
        x = np.linspace(XL[0], -C.CX, 300)
        up, lo = C.funnel_curves(x)
        ax.fill_between(x, lo, up, color=C.FUNNEL, alpha=0.08, lw=0, zorder=2.5)
        ax.plot(x, up, color=C.FUNNEL, lw=0.9, ls='--', zorder=3)
        ax.plot(x, lo, color=C.FUNNEL, lw=0.9, ls='--', zorder=3)
    T = C.trajectories(k)
    for s in FIVE:
        d = T[s]
        o = S[s][f'{k}_outcome']
        _, col, mk, ls = C.OUTCOME[o]
        ax.plot(d['xh'], d['yh'], color=col, lw=0.9, ls=ls, zorder=7, alpha=0.95)
        ax.plot(d['xh'][-1], d['yh'][-1], marker=mk, ms=7 if mk == '*' else 5, color=col,
                mec='white', mew=0.4, ls='', zorder=9)
    ax.plot(C.DP[0] + C.HP, C.DP[1], 'D', ms=3.8, color='#333333', mec='white', mew=0.4, zorder=10)
    ax.set_title(lab)
    ax.set_xlabel(r'$x_h$ [m]')
    ax.grid(alpha=0.35)
axs[0].set_ylabel(r'$y_h$ [m]')
handles = [Line2D([], [], color=C.DOCK, lw=1.0, marker='*', ms=7, label='dock'),
           Line2D([], [], color=C.COLL, lw=1.0, ls='--', marker='X', ms=5, label='collision'),
           Line2D([], [], color=C.TOUT, lw=1.0, ls='--', marker='s', ms=4.5, label='timeout'),
           Patch(fc=C.SET_FILL, ec=C.SET_LINE, lw=0.9, alpha=0.6, label='reach avoid set'),
           Patch(fc=C.FUNNEL, ec=C.FUNNEL, ls='--', alpha=0.25, label='funnel corridor'),
           Line2D([], [], ls='', marker='D', ms=3.8, color='#333333', label='DP point')]
fig.legend(handles=handles, loc='upper center', ncol=6, frameon=False,
           bbox_to_anchor=(0.5, 1.1), handlelength=1.8, columnspacing=1.1)
C.save(fig, 'trajectories_montecarlo')
