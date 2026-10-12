"""Candidate versions of Figs 5 to 8 on the finalfinal package, for choosing the baseline.

    python cand_figs.py      -> fig_src/preview/candidates/*.png   (manuscript untouched)

Fig 5   the new set (w095_ema_20k) at three headings, outline at tau, as the manuscript draws it
Fig 6   representative recovery of the proposed method, alpha_side 0.60 and 0.50
Fig 7   Monte Carlo summary per funnel option A, B, C: proposed vs the two gains on that funnel
Fig 8   head point paths of five seas per option, same five in every panel
Verdicts follow the package's scoring: dock, dock with a QP failure, contact inside the pocket
(not a collision by the agreed rule), collision outside the pocket, timeout.
"""
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import ff_common as F
import common as C

C.style()
F.OUTDIR.mkdir(parents=True, exist_ok=True)


PAPER = False      # True: manuscript figures into fig/ (no candidate titles, (1)/(2) labels)
SET_LEVEL = -0.006 # level the learned set is DRAWN at (author's choice 2026-10-11); the run handed over at C.TAU = -0.005


def save(fig, name, paper_name=None):
    if PAPER and paper_name:
        C.save(fig, paper_name)
        return
    fig.savefig(F.OUTDIR / f'{name}.png', dpi=200, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print('wrote', F.OUTDIR / f'{name}.png')


def title(fig, text, y):
    if not PAPER:
        fig.suptitle(text, fontsize=8, y=y, color=C.INK2)


def name_of(run, opt_runs):
    """Panel name: the manuscript calls the two gains RaCBF (1) and (2)."""
    if run.startswith('proposed'):
        return 'Proposed'
    if PAPER:
        return 'RaCBF (1)' if run == opt_runs[0] else 'RaCBF (2)'
    return F.label(run)


def short(run):
    return run.split('_as')[0].replace('_b', ' b').replace('_knee', ' knee').replace('_adock', ' adock')


# ------------------------------------------------------------------ Fig 5: the new set
def fig5():
    XL, YL = (-3.0, 0.9), (-0.6, 2.0)
    norm = TwoSlopeNorm(vmin=-0.04, vcenter=0.0, vmax=0.04)
    cmap = C.VCMAP.copy(); cmap.set_bad('#e9e9e9')
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, deg in zip(axs, (-10, 0, 10)):
        xs, ys, V, av, tg = C.value_slice(C.nominal_state(deg), XL, YL, h=C.AUTHOR_STEP)
        im = ax.pcolormesh(xs, ys, np.where(av, np.nan, np.clip(V, -0.04, 0.04)), cmap=cmap, norm=norm,
                           shading='auto', rasterized=True, zorder=0.5)
        C.draw_geometry(ax, XL, YL, wall=True, target=False)
        ax.add_patch(plt.Rectangle((-C.CX, -C.CY), 2 * C.CX, 2 * C.CY, fc='none', ec=C.SET_EDGE, lw=0.6,
                                   ls=':', zorder=5))
        for p in C.set_polys(xs, ys, V, av, SET_LEVEL, 'band' if deg <= 0 else 'author'):   # candidate Z1
            ax.plot(p[:, 0], p[:, 1], color=C.SET_LINE, lw=0.9, zorder=4)
        ax.set_title(r'$\psi_r = %+d^\circ$' % deg if deg else r'$\psi_r = 0^\circ$')
        ax.set_xlabel(r'$x_h$ [m]'); ax.set_xticks([-3, -2, -1, 0])
    axs[0].set_ylabel(r'$y_h$ [m]')
    cb = fig.colorbar(im, cax=fig.add_axes([0.915, 0.2, 0.012, 0.62]))
    cb.set_label('value $V$', fontsize=8.5); cb.set_ticks([-0.04, -0.02, 0, 0.02, 0.04]); cb.ax.tick_params(labelsize=7)
    fig.legend(handles=[Patch(fc=C.ARM, label='cradle'), Patch(fc=C.MS_FILL, ec=C.MS_EDGE, label='mothership'),
                        Patch(fc='#e9e9e9', ec='#9a9a9a', hatch='////', lw=0, label='approach wall'),
                        Line2D([], [], color=C.SET_LINE, lw=0.9, label=r'set boundary $V=0$')],
               loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    title(fig, 'Candidate Fig. 5   new set w095_ema_20k (outline at tau, labelled V = 0 as in the manuscript)', 1.02)
    save(fig, 'cand_fig5_set', 'reach_avoid_set')


# ------------------------------------------------------------------ Fig 6: representative recovery
def fig6(prop, partners, sea=None):
    res, ver = F.load(prop)
    bres = [F.load(b)[1] for b in partners]
    cand = [i for i, r in enumerate(res) if r['outcome'] == 'success' and r.get('t_switch') is not None]
    start = lambda i: res[i]['traj']['xr'][0] + 0.5 * res[i]['traj']['yr'][0]
    cand.sort(key=start)
    hard = [i for i in cand if any(b.get(i) not in ('success',) for b in bres)]
    k = (hard or cand)[0] if sea is None else sea
    d = res[k]['traj']; t = d['t']; th = res[k]['t_switch']
    V = C.value(np.stack([d['xr'], d['yr'], d['psi'], d['u'], d['v'], d['r'], d['delta'], d['F']], 1))
    ih = int(np.argmax(d['mode'] > 0.5))
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.55, 1.0], wspace=0.3, hspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    XL = (min(-3.0, d['xr'].min() - 0.4), 1.0); YL = (-0.6, max(1.2, d['yr'].max() + 0.35))
    C.draw_geometry(ax, XL, YL, wall=False, target=True)
    sh = (d['psi'][ih - 1], d['u'][ih - 1], d['v'][ih - 1], d['r'][ih - 1], d['delta'][ih - 1], d['F'][ih - 1])
    C.draw_set(ax, C.slice_set(sh, XL, YL, level=SET_LEVEL, method='band_wall'), alpha=0.35, z=2.5)
    ax.plot(d['xr'][:ih], d['yr'][:ih], color=C.PH1, lw=1.3, zorder=7)
    ax.plot(d['xr'][ih - 1:], d['yr'][ih - 1:], color=C.PH2, lw=1.3, zorder=7)
    for i in np.unique(np.r_[np.linspace(0, ih - 1, 4).astype(int), ih - 1, np.linspace(ih, len(t) - 1, 3).astype(int)]):
        col = C.PH1 if i < ih else C.PH2
        C.hull(ax, d['xr'][i], d['yr'][i], d['psi'][i], fc=col, ec=col, alpha=0.28, z=8)
        ax.plot(d['xh'][i], d['yh'][i], 'o', ms=2.6, color='#c2185b', zorder=9)
    ax.plot(d['xr'][ih - 1], d['yr'][ih - 1], 'o', ms=5, color='k', zorder=10)
    ax.plot(*C.DP, 'D', ms=4.5, color='#333333', mec='white', mew=0.5, zorder=10)
    ax.plot(0, 0, '*', ms=8, color='#b03a2e', zorder=10)
    ax.set_xlabel(r'$x_r$ [m]'); ax.set_ylabel(r'$y_r$ [m]'); ax.grid(alpha=0.35)
    ax.legend(handles=[Line2D([], [], color=C.PH1, lw=1.3, label='Phase 1'),
                       Line2D([], [], color=C.PH2, lw=1.3, label='Phase 2'),
                       Line2D([], [], ls='', marker='o', ms=5, color='k', label='handover'),
                       Line2D([], [], ls='', marker='o', ms=3, color='#c2185b', label='head point'),
                       Line2D([], [], ls='', marker='D', ms=4.5, color='#333333', label='DP point'),
                       Line2D([], [], ls='', marker='*', ms=8, color='#b03a2e', label='cradle center'),
                       Patch(fc=C.SET_FILL, ec=C.SET_LINE, lw=0.9, alpha=0.6, label='set at handover')],
              loc='upper right', fontsize=7, framealpha=0.95, edgecolor='#cccccc', handlelength=1.4,
              borderpad=0.4, labelspacing=0.3)
    # the value panel is re-based on the hand over threshold (author's request 2026-10-12): V - tau,
    # so the hand over happens where the curve crosses zero
    series = [(V - C.TAU, r'$V$', None), (d['F'], r'$F$ [N]', (-3.0, 9.0)),
              (np.degrees(0.01 * d['delta']), r'$\delta$ [deg]', (-30.0, 30.0)), (d['avoid'], 'clearance [m]', None)]
    axes = []
    for j, (y, lab, lim) in enumerate(series):
        a = fig.add_subplot(gs[j, 1], sharex=axes[0] if axes else None); axes.append(a)
        a.axvspan(t[0], th, color=C.PH1, alpha=0.06, lw=0); a.axvspan(th, t[-1], color=C.PH2, alpha=0.06, lw=0)
        a.axvline(th, color='#777777', lw=0.7, ls='--')
        a.plot(t, y, color='#1b7837' if j in (0, 3) else (C.PH1 if j == 2 else '#b03a2e'), lw=1.1)
        for v in (lim or ()):
            a.axhline(v, color='#999999', lw=0.6, ls=':')
        a.set_ylabel(lab, fontsize=7); a.grid(alpha=0.35); a.tick_params(labelsize=7)
        if j < 3:
            a.tick_params(labelbottom=False)
    axes[0].axhline(0.0, color='#555555', lw=0.6, ls='--')   # hand over threshold, V re-based so tau sits at 0
    axes[3].axhline(0.0, color=C.COLL, lw=0.6, ls=':'); axes[3].set_ylim(bottom=-0.05)
    axes[3].set_xlabel('time [s]'); axes[3].set_xlim(t[0], t[-1]); fig.align_ylabels(axes)
    tag = ', '.join(f'{b[0]} a{F.alpha(b):g} {bres[n].get(k)}' for n, b in enumerate(partners))
    title(fig, f'Candidate Fig. 6   {prop}, sea {k}\nbaselines on this sea: {tag}', 1.04)
    save(fig, f'cand_fig6_rep_{prop}' + ('' if sea is None else f'_sea{sea}'), 'rep_run')


# ------------------------------------------------------------------ Fig 7: Monte Carlo summary
def fig7(opt):
    collide, timeout, prop = F.RUNS[opt]
    runs = [prop, collide, timeout]
    names = (['Proposed', 'RaCBF\n(1)', 'RaCBF\n(2)'] if PAPER else
             ['Proposed'] + ['RaCBF\n' + r'$\alpha_{dock}$' + f'={F.alpha(r):g}' for r in (collide, timeout)])
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 1.75), gridspec_kw=dict(wspace=0.38))
    rng = np.random.default_rng(3)
    ax = axs[0]
    for i, run in enumerate(runs):
        res, ver = F.load(run)
        b = 0
        for v in F.ORDER:
            n = sum(1 for x in ver.values() if x == v)
            if not n:
                continue
            lab, col, _, _, hatch = F.VERDICT[v]
            ax.bar(i, n, bottom=b, width=0.62, color=col, ec='white', lw=0.8, hatch=hatch)
            if v == 'success':
                ax.text(i, b + n / 2, str(n), ha='center', va='center', color='white', fontsize=8, fontweight='bold')
            else:
                ax.text(i + 0.36, b + n / 2, str(n), ha='left', va='center', color=C.INK, fontsize=7, fontweight='bold')
            b += n
    ax.set_ylim(0, 102); ax.set_xlim(-0.55, 2.75); ax.set_ylabel('seas out of 100'); ax.set_title('(a) Recovery outcome')
    ax = axs[1]
    for i, run in enumerate(runs):
        res, ver = F.load(run)
        tt = np.array([r['t_end'] for j, r in enumerate(res) if ver[j] in ('success', 'success_qp')])
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(tt)), tt, s=5, color=C.DOCK, alpha=0.45, lw=0)
        ax.errorbar(i + 0.32, tt.mean(), yerr=tt.std(), fmt='o', ms=3.5, color=C.INK, capsize=2.5, lw=0.9)
        ax.text(i + 0.32, tt.mean() + tt.std() + 1.2, f'{tt.mean():.1f}', ha='center', va='bottom', fontsize=7)
    ax.set_xlim(-0.5, 2.65); ax.set_ylim(0, None); ax.set_ylabel('docking time [s]'); ax.set_title('(b) Docking time')
    ax = axs[2]
    for i, run in enumerate(runs):
        res, ver = F.load(run)
        for v in F.ORDER:
            c = np.array([r['clear_before_entry'] for j, r in enumerate(res) if ver[j] == v])
            if len(c):
                _, col, mk, _, _ = F.VERDICT[v]
                ax.scatter(i + rng.uniform(-0.22, 0.22, len(c)), c, s=7 if v == 'success' else 12,
                           marker='o' if v.startswith('success') else mk, color=col,
                           alpha=0.5 if v == 'success' else 0.85, lw=0, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [min(r['clear_before_entry'] for r in res)] * 2, color=C.INK, lw=1.0, zorder=4)
    ax.axhline(0, color=C.INK, lw=0.7, ls=':'); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel('min. clearance before entry [m]', fontsize=7.5); ax.set_title('(c) Clearance to obstacles')
    for a in axs:
        a.set_xticks(range(3)); a.set_xticklabels(names, fontsize=6.2); a.grid(axis='y', alpha=0.5)
    present = [v for v in F.ORDER if any(v in F.load(r)[1].values() for r in runs)]
    fig.legend(handles=[Patch(fc=F.VERDICT[v][1], ec='white', hatch=F.VERDICT[v][4], label=F.VERDICT[v][0])
                        for v in present] + [Line2D([], [], color=C.INK, lw=1.0, label='worst run')],
               loc='lower center', ncol=len(present) + 1, frameon=False, bbox_to_anchor=(0.5, -0.36))
    title(fig, f'Candidate Fig. 7   option {opt}: {short(collide)} | {short(timeout)} | {prop}', 1.08)
    save(fig, f'cand_fig7_option{opt}', 'montecarlo_summary')


# ------------------------------------------------------------------ Fig 8: trajectories
def pick_five(opt):
    collide, timeout, prop = F.RUNS[opt]
    rows = {(r['run'], int(r['trial'])): r for r in csv.DictReader(open(F.FF / 'outcomes.csv'))}
    vc, vt = F.load(collide)[1], F.load(timeout)[1]
    out = sorted([i for i, v in vc.items() if v == 'outside'], key=lambda i: abs(float(rows[(collide, i)]['hull_stopped_short_m'])))
    tmo = [i for i, v in vt.items() if v == 'timeout' and vc[i] == 'success' and i not in out]
    tmo.sort(key=lambda i: float(rows[(timeout, i)]['hull_stopped_short_m']))
    clean = [i for i in range(100) if vc[i] == 'success' and vt[i] == 'success']
    five = out[:2] + tmo[:2] + clean[:1]
    if len(five) < 5:
        five += [i for i in range(100) if i not in five][:5 - len(five)]
    return five


def fig8(opt, five=None, XL=(-4.6, 1.0), YL=(-0.55, 2.5), height=2.05):
    collide, timeout, prop = F.RUNS[opt]
    five = five or pick_five(opt)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, height), sharey=True, gridspec_kw=dict(wspace=0.05))
    for ax, run in zip(axs, (prop, collide, timeout)):
        C.draw_geometry(ax, XL, YL, wall=False, target=True)
        res, ver = F.load(run)
        if run.startswith('proposed'):
            C.draw_set(ax, C.boundary(0, SET_LEVEL, method='band'), alpha=0.4, z=2.5)
        else:
            F.set_funnel(run)
            x = np.linspace(XL[0], -C.CX, 300); up, lo = C.funnel_curves(x)
            ax.fill_between(x, lo, up, color=C.FUNNEL, alpha=0.08, lw=0, zorder=2.5)
            ax.plot(x, up, color=C.FUNNEL, lw=0.9, ls='--', zorder=3); ax.plot(x, lo, color=C.FUNNEL, lw=0.9, ls='--', zorder=3)
        for s in five:
            d = res[s]['traj']; _, col, mk, ls, _ = F.VERDICT[ver[s]]
            ax.plot(d['xh'], d['yh'], color=col, lw=0.9, ls=ls, zorder=7)
            ax.plot(d['xh'][-1], d['yh'][-1], marker=mk, ms=7 if mk == '*' else 5, color=col, mec='white', mew=0.4,
                    ls='', zorder=9)
        ax.plot(C.DP[0] + C.HP, C.DP[1], 'D', ms=3.8, color='#333333', mec='white', mew=0.4, zorder=10)
        ax.set_title(name_of(run, F.RUNS[opt]), fontsize=8.5); ax.set_xlabel(r'$x_h$ [m]'); ax.grid(alpha=0.35)
    axs[0].set_ylabel(r'$y_h$ [m]')
    hs = [Line2D([], [], color=F.VERDICT[v][1], lw=1.0, ls=F.VERDICT[v][3], marker=F.VERDICT[v][2], ms=5,
                 label=F.VERDICT[v][0]) for v in ('success', 'outside', 'timeout')]
    hs += [Patch(fc=C.SET_FILL, ec=C.SET_LINE, lw=0.9, alpha=0.6, label='reach avoid set'),
           Patch(fc=C.FUNNEL, ec=C.FUNNEL, alpha=0.25, label='funnel corridor'),
           Line2D([], [], ls='', marker='D', ms=3.8, color='#333333', label='DP point')]
    fig.legend(handles=hs, loc='upper center', ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.97 + 0.15 / height),
               handlelength=1.6, columnspacing=0.9, fontsize=7)
    title(fig, f'Candidate Fig. 8   option {opt}, seas {five}', 1.2)
    save(fig, f'cand_fig8_option{opt}', 'trajectories_montecarlo')


if __name__ == '__main__':
    fig5()
    fig6('proposed_as0.60', [F.RUNS['A'][0], F.RUNS['A'][1], F.RUNS['B'][0], F.RUNS['B'][1]])
    fig6('proposed_as0.50', [F.RUNS['C'][0], F.RUNS['C'][1]])
    for opt in 'ABC':
        fig7(opt)
        fig8(opt)
