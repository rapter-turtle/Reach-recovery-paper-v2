"""Representative recovery of the proposed two phase method.

Scenario choice follows the study's own rule (fig678_mc.py): docked runs that hand over,
sorted to show the longest approach (start farthest astern and closest to the cradle line),
preferring a scenario that one of the baselines fails.  V is recomputed from the exported
model on every logged state, so the curve is aligned with the state it belongs to.
"""
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common as C

C.style()
S = C.scenarios()
cand = [r for r in S if r['prop_outcome'] == 'success' and r['prop_t_handover']]
cand.sort(key=lambda r: float(r['start_x_rel']) + 0.5 * float(r['start_y_rel']))
hard = [r for r in cand if r['racbf1_outcome'] != 'success' or r['racbf2_outcome'] != 'success']
row = (hard or cand)[0]
k = int(row['scenario'])
th = float(row['prop_t_handover'])
print('scenario', k, 'handover', th, 'racbf1', row['racbf1_outcome'], 'racbf2', row['racbf2_outcome'])

d = C.trajectories('prop')[k]
t = d['t']
V = C.value(C.rel_states(d))
ph2 = d['mode'] > 0.5
ih = int(np.argmax(ph2))                    # first Phase 2 row
iend = len(t) - 1

fig = plt.figure(figsize=(7.2, 3.0))
gs = fig.add_gridspec(4, 2, width_ratios=[1.55, 1.0], wspace=0.3, hspace=0.18)
ax = fig.add_subplot(gs[:, 0])
XL = (min(-3.0, d['xr'].min() - 0.4), 1.0)
YL = (-0.75, max(1.2, d['yr'].max() + 0.35))
C.draw_geometry(ax, XL, YL, wall=False, target=True)

# learned set at the handover state, in head point coordinates
sh = (d['psi'][ih - 1], d['u'][ih - 1], d['v'][ih - 1], d['r'][ih - 1], d['delta'][ih - 1], d['F'][ih - 1])
C.draw_set(ax, C.slice_set(sh, (XL[0], XL[1]), (YL[0], YL[1]), level=C.TAU), alpha=0.35, z=2.5)

ax.plot(d['xr'][:ih], d['yr'][:ih], color=C.PH1, lw=1.3, zorder=7)
ax.plot(d['xr'][ih - 1:], d['yr'][ih - 1:], color=C.PH2, lw=1.3, zorder=7)
for i in np.unique(np.r_[np.linspace(0, ih - 1, 4).astype(int), ih - 1,
                        np.linspace(ih, iend, 3).astype(int)]):
    col = C.PH1 if i < ih else C.PH2
    C.hull(ax, d['xr'][i], d['yr'][i], d['psi'][i], fc=col, ec=col, alpha=0.28, z=8)
    ax.plot(d['xh'][i], d['yh'][i], 'o', ms=2.6, color='#c2185b', zorder=9)
ax.plot(d['xr'][ih - 1], d['yr'][ih - 1], 'o', ms=5, color='k', zorder=10)
ax.plot(*C.DP, 'D', ms=4.5, color='#333333', mec='white', mew=0.5, zorder=10)
ax.plot(0, 0, '*', ms=8, color='#b03a2e', zorder=10)
ax.set_xlabel(r'$x_r$ [m]')
ax.set_ylabel(r'$y_r$ [m]')
ax.grid(alpha=0.35)
leg = [Line2D([], [], color=C.PH1, lw=1.3, label='Phase 1'),
       Line2D([], [], color=C.PH2, lw=1.3, label='Phase 2'),
       Line2D([], [], ls='', marker='o', ms=5, color='k', label='handover'),
       Line2D([], [], ls='', marker='o', ms=3, color='#c2185b', label='head point'),
       Line2D([], [], ls='', marker='D', ms=4.5, color='#333333', label='DP point'),
       Line2D([], [], ls='', marker='*', ms=8, color='#b03a2e', label='cradle center'),
       Patch(fc=C.SET_FILL, ec=C.SET_LINE, lw=0.9, alpha=0.6, label='set at handover')]
ax.legend(handles=leg, loc='upper right', fontsize=7, frameon=True, framealpha=0.95,
          edgecolor='#cccccc', handlelength=1.4, borderpad=0.4, labelspacing=0.3)

# time histories
dl = np.degrees(C.DELTA_SCALE * d['delta'])
series = [(V, r'$V$', None), (d['F'], r'$F$ [N]', (-3.0, 9.0)),
          (dl, r'$\delta$ [deg]', (-30.0, 30.0)), (d['avoid'], 'clearance\n[m]', None)]
axes = []
for j, (y, lab, lim) in enumerate(series):
    a = fig.add_subplot(gs[j, 1], sharex=axes[0] if axes else None)
    axes.append(a)
    a.axvspan(t[0], th, color=C.PH1, alpha=0.06, lw=0)
    a.axvspan(th, t[-1], color=C.PH2, alpha=0.06, lw=0)
    a.axvline(th, color='#777777', lw=0.7, ls='--')
    a.plot(t, y, color='#1b7837' if j in (0, 3) else (C.PH1 if j == 2 else '#b03a2e'), lw=1.1)
    if lim:
        for v in lim:
            a.axhline(v, color='#999999', lw=0.6, ls=':')
    a.set_ylabel(lab, fontsize=8)
    a.grid(alpha=0.35)
    a.tick_params(labelsize=7)
    if j < 3:
        a.tick_params(labelbottom=False)
axes[0].axhline(0.0, color='#555555', lw=0.6, ls=':')
axes[0].axhline(C.TAU, color='#555555', lw=0.6, ls='--')
axes[3].axhline(0.0, color=C.COLL, lw=0.6, ls=':')
axes[3].set_ylim(bottom=-0.05)
axes[3].set_xlabel('time [s]')
axes[3].set_xlim(t[0], t[-1])
axes[0].text(0.5 * th, 1.02, 'Phase 1', transform=axes[0].get_xaxis_transform(), ha='center',
             va='bottom', fontsize=7.5, color=C.PH1)
axes[0].text(0.5 * (th + t[-1]), 1.02, 'Phase 2', transform=axes[0].get_xaxis_transform(),
             ha='center', va='bottom', fontsize=7.5, color=C.PH2)
fig.align_ylabels(axes)
C.save(fig, 'rep_run')
