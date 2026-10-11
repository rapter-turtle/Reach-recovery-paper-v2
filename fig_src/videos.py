"""Videos of the representative recovery (Fig. 6) and the five sea comparison (Fig. 8).

    python videos.py   -> <project>/videos/rep_run.mp4, trajectories_montecarlo.mp4

Same data, design and set outlines (candidate Z1) as the manuscript figures, animated.  The logged
states (0.25 s) are interpolated linearly to the frame rate.  Written outside new_CEP_draft so the
videos never sync to Overleaf.
"""
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

import cand_figs as K
import ff_common as F
import common as C

C.style()
K.PAPER = True          # manuscript panel names: Proposed, RaCBF (1), RaCBF (2)
OUT = F.FF.parent / 'videos'
OUT.mkdir(exist_ok=True)
FPS = 30


def hull_xy(cx, cy, psi):
    hp, beam, aft = C.HP, 0.125, C.SP
    pts = np.array([[hp, 0.0], [hp - 0.05, 0.07], [hp - 0.16, beam], [-aft, beam * 0.85],
                    [-aft, -beam * 0.85], [hp - 0.16, -beam], [hp - 0.05, -0.07]])
    c, s = math.cos(psi), math.sin(psi)
    return pts @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])


def interp(d, t):
    """State at time t (clamped to the run), psi unwrapped before interpolating."""
    tt = d['t']
    t = min(max(t, tt[0]), tt[-1])
    out = {k: float(np.interp(t, tt, d[k])) for k in ('xr', 'yr', 'xh', 'yh')}
    out['psi'] = float(np.interp(t, tt, np.unwrap(d['psi'])))
    return out


# ------------------------------------------------------------------ representative recovery
def rep_video(prop='proposed_as0.60', sea=28, hold=1.5):
    res, _ = F.load(prop)
    r = res[sea]; d = r['traj']; t = d['t']; th = r['t_switch']
    V = C.value(np.stack([d['xr'], d['yr'], d['psi'], d['u'], d['v'], d['r'], d['delta'], d['F']], 1))
    ih = int(np.argmax(d['mode'] > 0.5))
    XL = (min(-3.0, d['xr'].min() - 0.4), 1.0); YL = (-0.6, max(1.2, d['yr'].max() + 0.35))
    sh = (d['psi'][ih - 1], d['u'][ih - 1], d['v'][ih - 1], d['r'][ih - 1], d['delta'][ih - 1], d['F'][ih - 1])
    set_polys = C.slice_set(sh, XL, YL, level=K.SET_LEVEL, method='band')

    fig = plt.figure(figsize=(7.2, 3.0), dpi=200)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.55, 1.0], wspace=0.32, hspace=0.2,
                          left=0.07, right=0.985, top=0.93, bottom=0.14)
    ax = fig.add_subplot(gs[:, 0])
    C.draw_geometry(ax, XL, YL, wall=False, target=True)
    ax.plot(*C.DP, 'D', ms=4.5, color='#333333', mec='white', mew=0.5, zorder=10)
    ax.plot(0, 0, '*', ms=8, color='#b03a2e', zorder=10)
    ax.set_xlabel(r'$x_r$ [m]'); ax.set_ylabel(r'$y_r$ [m]'); ax.grid(alpha=0.35)
    set_art = []
    for p in set_polys:
        set_art.append(ax.add_patch(Polygon(p, closed=True, fc=C.SET_FILL, ec='none', alpha=0.35, zorder=2.5,
                                            visible=False)))
        set_art.append(ax.plot(p[:, 0], p[:, 1], color=C.SET_LINE, lw=0.9, zorder=2.6, visible=False)[0])
    tr1, = ax.plot([], [], color=C.PH1, lw=1.3, zorder=7)
    tr2, = ax.plot([], [], color=C.PH2, lw=1.3, zorder=7)
    hull = ax.add_patch(Polygon(hull_xy(d['xr'][0], d['yr'][0], d['psi'][0]), closed=True, fc=C.PH1,
                                ec=C.PH1, alpha=0.45, lw=0.8, zorder=8))
    head, = ax.plot([], [], 'o', ms=3, color='#c2185b', zorder=9)
    hand, = ax.plot([], [], 'o', ms=5, color='k', zorder=10)
    clock = ax.text(0.02, 0.26, '', transform=ax.transAxes, fontsize=8, va='top', color=C.INK)
    phase = ax.text(0.02, 0.18, '', transform=ax.transAxes, fontsize=8, va='top', fontweight='bold')
    ax.legend(handles=[Line2D([], [], color=C.PH1, lw=1.3, label='Phase 1'),
                       Line2D([], [], color=C.PH2, lw=1.3, label='Phase 2'),
                       Line2D([], [], ls='', marker='o', ms=5, color='k', label='handover'),
                       Line2D([], [], ls='', marker='o', ms=3, color='#c2185b', label='head point'),
                       Line2D([], [], ls='', marker='D', ms=4.5, color='#333333', label='DP point'),
                       Line2D([], [], ls='', marker='*', ms=8, color='#b03a2e', label='cradle center'),
                       Patch(fc=C.SET_FILL, ec=C.SET_LINE, lw=0.9, alpha=0.6, label='set at handover')],
              loc='upper right', fontsize=6.5, framealpha=0.95, edgecolor='#cccccc', handlelength=1.4,
              borderpad=0.4, labelspacing=0.3)
    series = [(V, r'$V$', None), (d['F'], r'$F$ [N]', (-3.0, 9.0)),
              (np.degrees(0.01 * d['delta']), r'$\delta$ [deg]', (-30.0, 30.0)), (d['avoid'], 'clearance\n[m]', None)]
    lines, cursors, axes = [], [], []
    for j, (y, lab, lim) in enumerate(series):
        a = fig.add_subplot(gs[j, 1], sharex=axes[0] if axes else None); axes.append(a)
        a.axvspan(t[0], th, color=C.PH1, alpha=0.06, lw=0); a.axvspan(th, t[-1], color=C.PH2, alpha=0.06, lw=0)
        for v in (lim or ()):
            a.axhline(v, color='#999999', lw=0.6, ls=':')
        lo, hi = float(np.min(y)), float(np.max(y))
        if lim:
            lo, hi = min(lo, lim[0]), max(hi, lim[1])
        pad = 0.08 * (hi - lo + 1e-9)
        a.set_ylim(lo - pad, hi + pad)
        lines.append(a.plot([], [], color='#1b7837' if j in (0, 3) else (C.PH1 if j == 2 else '#b03a2e'), lw=1.1)[0])
        cursors.append(a.axvline(t[0], color='#777777', lw=0.7, ls='--'))
        a.set_ylabel(lab, fontsize=8); a.grid(alpha=0.35); a.tick_params(labelsize=7)
        if j < 3:
            a.tick_params(labelbottom=False)
    axes[0].axhline(0.0, color='#555555', lw=0.6, ls=':'); axes[0].axhline(C.TAU, color='#555555', lw=0.6, ls='-.')
    axes[3].axhline(0.0, color=C.COLL, lw=0.6, ls=':')
    axes[3].set_xlabel('time [s]'); axes[3].set_xlim(t[0], t[-1]); fig.align_ylabels(axes)

    n = int(round((t[-1] - t[0]) * FPS)) + int(hold * FPS)
    writer = FFMpegWriter(fps=FPS, bitrate=4000)
    with writer.saving(fig, str(OUT / 'rep_run.mp4'), dpi=200):
        for f in range(n):
            tc = min(t[0] + f / FPS, t[-1])
            k = int(np.searchsorted(t, tc, side='right'))
            s = interp(d, tc)
            p1 = t <= min(tc, th)
            tr1.set_data(np.r_[d['xr'][p1], s['xr'] if tc <= th else []], np.r_[d['yr'][p1], s['yr'] if tc <= th else []])
            if tc > th:
                p2 = (t >= t[ih - 1]) & (t <= tc)
                tr2.set_data(np.r_[d['xr'][p2], s['xr']], np.r_[d['yr'][p2], s['yr']])
            col = C.PH1 if tc <= th else C.PH2
            hull.set_xy(hull_xy(s['xr'], s['yr'], s['psi'])); hull.set_facecolor(col); hull.set_edgecolor(col)
            head.set_data([s['xh']], [s['yh']])
            if tc >= th:
                hand.set_data([d['xr'][ih - 1]], [d['yr'][ih - 1]])
                for art in set_art:
                    art.set_visible(True)
            clock.set_text(f't = {tc:5.2f} s')
            phase.set_text('Phase 1  DOB-MPC with RaCBF' if tc < th else 'Phase 2  reach avoid policy')
            phase.set_color(col)
            for j, (y, _, _) in enumerate(series):
                lines[j].set_data(np.r_[t[:k], tc], np.r_[y[:k], np.interp(tc, t, y)])
                cursors[j].set_xdata([tc, tc])
            writer.grab_frame()
    plt.close(fig)
    print('wrote', OUT / 'rep_run.mp4', f'({n} frames)')


# ------------------------------------------------------------------ five sea comparison
def five_video(opt='A', five=(48, 12, 92, 76, 10), XL=(-5.6, 1.0), YL=(-0.6, 4.2), speed=6.0, hold=2.0):
    collide, timeout, prop = F.RUNS[opt]
    runs = (prop, collide, timeout)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), dpi=200, sharey=True,
                            gridspec_kw=dict(wspace=0.05, left=0.07, right=0.99, top=0.74, bottom=0.17))
    arts = []
    tmax = 0.0
    for ax, run in zip(axs, runs):
        C.draw_geometry(ax, XL, YL, wall=False, target=True)
        res, ver = F.load(run)
        if run.startswith('proposed'):
            C.draw_set(ax, C.boundary(0, K.SET_LEVEL, method='band'), alpha=0.4, z=2.5)
        else:
            F.set_funnel(run)
            x = np.linspace(XL[0], -C.CX, 300); up, lo = C.funnel_curves(x)
            ax.fill_between(x, lo, up, color=C.FUNNEL, alpha=0.08, lw=0, zorder=2.5)
            ax.plot(x, up, color=C.FUNNEL, lw=0.9, ls='--', zorder=3); ax.plot(x, lo, color=C.FUNNEL, lw=0.9, ls='--', zorder=3)
        ax.plot(C.DP[0] + C.HP, C.DP[1], 'D', ms=3.8, color='#333333', mec='white', mew=0.4, zorder=10)
        ax.set_title(K.name_of(run, F.RUNS[opt]), fontsize=8.5); ax.set_xlabel(r'$x_h$ [m]'); ax.grid(alpha=0.35)
        for s in five:
            d = res[s]['traj']; _, col, mk, ls, _ = F.VERDICT[ver[s]]
            tmax = max(tmax, d['t'][-1])
            trail, = ax.plot([], [], color=col, lw=0.9, ls=ls, zorder=7)
            h = ax.add_patch(Polygon(hull_xy(d['xr'][0], d['yr'][0], d['psi'][0]), closed=True, fc=col, ec=col,
                                     alpha=0.5, lw=0.6, zorder=8))
            end, = ax.plot([], [], marker=mk, ms=7 if mk == '*' else 5, color=col, mec='white', mew=0.4, ls='',
                           zorder=9)
            arts.append((d, trail, h, end))
    axs[0].set_ylabel(r'$y_h$ [m]')
    clock = fig.text(0.5, 0.995, '', ha='center', va='top', fontsize=8.5, color=C.INK)
    hs = [Line2D([], [], color=F.VERDICT[v][1], lw=1.0, ls=F.VERDICT[v][3], marker=F.VERDICT[v][2], ms=5,
                 label=F.VERDICT[v][0]) for v in ('success', 'outside', 'timeout')]
    hs += [Patch(fc=C.SET_FILL, ec=C.SET_LINE, lw=0.9, alpha=0.6, label='reach avoid set'),
           Patch(fc=C.FUNNEL, ec=C.FUNNEL, alpha=0.25, label='funnel corridor'),
           Line2D([], [], ls='', marker='D', ms=3.8, color='#333333', label='DP point')]
    fig.legend(handles=hs, loc='upper center', ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.93),
               handlelength=1.6, columnspacing=0.9, fontsize=6.8)

    n = int(round(tmax / speed * FPS)) + int(hold * FPS)
    writer = FFMpegWriter(fps=FPS, bitrate=4000)
    with writer.saving(fig, str(OUT / 'trajectories_montecarlo.mp4'), dpi=200):
        for f in range(n):
            tc = min(f * speed / FPS, tmax)
            for d, trail, h, end in arts:
                tt = d['t']; k = int(np.searchsorted(tt, tc, side='right'))
                s = interp(d, tc)
                trail.set_data(np.r_[d['xh'][:k], s['xh']], np.r_[d['yh'][:k], s['yh']])
                h.set_xy(hull_xy(s['xr'], s['yr'], s['psi']))
                if tc >= tt[-1]:
                    end.set_data([d['xh'][-1]], [d['yh'][-1]])
            clock.set_text(f't = {tc:5.1f} s   (played at {speed:g}x)')
            writer.grab_frame()
    plt.close(fig)
    print('wrote', OUT / 'trajectories_montecarlo.mp4', f'({n} frames)')


if __name__ == '__main__':
    rep_video()
    five_video()
