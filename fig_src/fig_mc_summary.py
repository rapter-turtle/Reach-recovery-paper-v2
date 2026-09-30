"""Monte Carlo summary over the 150 sea scenarios.

(a) outcome counts, (b) docking time over the docked runs with the mean and one standard
deviation, (c) minimum clearance of every run to the obstacles, coloured by outcome.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import common as C

C.style()
S = C.scenarios()
N = len(S)
rng = np.random.default_rng(3)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw=dict(wspace=0.36))
xpos = np.arange(len(C.CTRL))
names = [lab.replace(' (', '\n(') for _, lab in C.CTRL]

# (a) outcomes
ax = axs[0]
for i, (k, _) in enumerate(C.CTRL):
    cnt = {o: sum(r[f'{k}_outcome'] == o for r in S) for o in ('success', 'crash', 'incomplete')}
    bottom = 0
    for o in ('success', 'crash', 'incomplete'):
        n = cnt[o]
        if n:
            ax.bar(i, n, bottom=bottom, width=0.62, color=C.OUTCOME[o][1], ec='white', lw=0.8)
            if o == 'success':
                ax.text(i, bottom + n / 2, str(n), ha='center', va='center', color='white',
                        fontsize=8.5, fontweight='bold')
            else:
                ax.text(i + 0.36, bottom + n / 2, str(n), ha='left', va='center', color=C.INK,
                        fontsize=7.5, fontweight='bold')
        bottom += n
ax.set_ylim(0, N * 1.02)
ax.set_xlim(-0.55, len(C.CTRL) - 0.25)
ax.set_ylabel(f'scenarios out of {N}')
ax.set_title('(a) Recovery outcome')

# (b) docking time
ax = axs[1]
for i, (k, _) in enumerate(C.CTRL):
    tt = np.array([float(r[f'{k}_t_end']) for r in S if r[f'{k}_outcome'] == 'success'])
    ax.scatter(i + rng.uniform(-0.2, 0.2, len(tt)), tt, s=5, color=C.DOCK, alpha=0.45, lw=0)
    m, sd = tt.mean(), tt.std()
    ax.errorbar(i + 0.32, m, yerr=sd, fmt='o', ms=3.5, color=C.INK, capsize=2.5, lw=0.9)
    ax.text(i + 0.32, m + sd + 1.2, f'{m:.1f}', ha='center', va='bottom', fontsize=7.5, color=C.INK)
ax.set_xlim(-0.5, len(C.CTRL) - 0.35)
ax.set_ylim(0, None)
ax.set_ylabel('docking time [s]')
ax.set_title('(b) Docking time (docked runs)')

# (c) minimum clearance of every run
ax = axs[2]
for i, (k, _) in enumerate(C.CTRL):
    for o in ('success', 'incomplete', 'crash'):
        c = np.array([float(r[f'{k}_min_clearance']) for r in S if r[f'{k}_outcome'] == o])
        if len(c):
            _, col, mk, _ = C.OUTCOME[o]
            ax.scatter(i + rng.uniform(-0.22, 0.22, len(c)), c, s=7 if o == 'success' else 12,
                       marker='o' if o == 'success' else mk, color=col,
                       alpha=0.5 if o == 'success' else 0.85, lw=0, zorder=3)
    w = min(float(r[f'{k}_min_clearance']) for r in S)
    ax.plot([i - 0.3, i + 0.3], [w, w], color=C.INK, lw=1.0, zorder=4)
ax.axhline(0.0, color=C.INK, lw=0.7, ls=':', zorder=2)
ax.text(-0.47, -0.008, 'contact', ha='left', va='top', fontsize=7, color=C.INK2)
ax.set_xlim(-0.5, len(C.CTRL) - 0.5)
ax.set_ylabel('minimum clearance [m]')
ax.set_title('(c) Clearance to obstacles')

for ax in axs:
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=7.5)
    ax.grid(axis='y', alpha=0.5)
handles = [Line2D([], [], ls='', marker='s', ms=6, color=C.DOCK, label='dock'),
           Line2D([], [], ls='', marker='X', ms=6, color=C.COLL, label='collision'),
           Line2D([], [], ls='', marker='s', ms=6, color=C.TOUT, label='timeout'),
           Line2D([], [], color=C.INK, lw=1.0, label='worst run'),
           Line2D([], [], ls='', marker='o', ms=3.5, color=C.INK, label=r'mean $\pm$ one std')]
fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
           bbox_to_anchor=(0.5, -0.2), handlelength=1.4, columnspacing=1.4)
C.save(fig, 'montecarlo_summary')
