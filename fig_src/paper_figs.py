"""Manuscript Figs 5 to 8 on the finalfinal package, option A (chosen 2026-10-11).

    python paper_figs.py      -> ../fig/{reach_avoid_set, rep_run, montecarlo_summary,
                                         trajectories_montecarlo}.pdf
    python fig_geometry.py    -> ../fig/set_geometry.pdf

Option A: funnel b 3.0, knee -1.188; RaCBF (1) alpha_dock 50, RaCBF (2) alpha_dock 0.8, both at
alpha_side 0.60, against proposed_as0.60 on the same 100 seas.  A contact after the head point
has entered the pocket counts as a dock (the agreed rule).
"""
import cand_figs as K
import ff_common as F

K.PAPER = True
K.fig5()
K.fig6('proposed_as0.60', [F.RUNS['A'][0], F.RUNS['A'][1]], sea=94)   # chosen 2026-10-12
K.fig7('A')
# seas: 48 RaCBF (1) strikes the outer corner of an arm at the mouth, 12 RaCBF (1) strikes the
# mothership side, 92 and 76 RaCBF (2) stalls astern of the mouth, 10 all three dock
K.fig8('A', five=[48, 12, 92, 76, 10], XL=(-5.6, 1.0), YL=(-0.6, 4.2), height=2.45)
