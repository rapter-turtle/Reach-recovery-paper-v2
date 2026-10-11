"""Candidate figures on the finalfinal package (6 baselines + 2 proposed, new set model).

Does NOT touch the manuscript figures: it reuses the drawing helpers of common.py with the
finalfinal geometry patched in at import, and writes only to fig_src/preview/candidates/.
"""
import csv
import pickle
import re
from pathlib import Path

import numpy as np

import common as C
from ff_model import Model

FF = Path(__file__).resolve().parents[2] / 'finalfinal'
OUTDIR = C.PREVIEW / 'candidates'

# finalfinal geometry (config.json, code/dynamics/ship_pure_ra.py)
C.DATA = FF
C.TW = 0.075
C.Y_MS = -0.3375
C.Y_WALL = 0.25
C.DP = (-1.82, 0.0)
_MODEL = Model(FF / 'model' / 'w095_ema_20k.pt')
C.value = _MODEL.value

RUNS = {
    'A': ('A_b3.0_knee-1.188_adock50_as0.60_collide', 'A_b3.0_knee-1.188_adock0.8_as0.60_timeout',
          'proposed_as0.60'),
    'B': ('B_b4.9_knee-1.155_adock50_as0.60_collide', 'B_b4.9_knee-1.155_adock1.2_as0.60_timeout',
          'proposed_as0.60'),
    'C': ('C_b4.9_knee-1.400_adock70_as0.50_collide', 'C_b4.9_knee-1.400_adock1.2_as0.50_timeout',
          'proposed_as0.50'),
}
VERDICT = {   # verdict -> (label, colour, marker, linestyle, hatch)
    'success': ('dock', C.DOCK, '*', '-', None),
    'success_qp': ('dock, QP failure', C.DOCK, '*', '-', '////'),
    'pocket': ('contact in pocket', '#6b6b6b', 'P', '--', None),
    'outside': ('collision', C.COLL, 'X', '--', None),
    'timeout': ('timeout', C.TOUT, 's', '--', None),
}
ORDER = ['success', 'success_qp', 'pocket', 'outside', 'timeout']


def gains(run):
    if run.startswith('proposed'):
        return None
    cbf = next(r['cbf'] for r in csv.DictReader(open(FF / 'summary.csv')) if r['run'] == run)
    return dict(re.findall(r'(\w+)=(-?[0-9.]+)', cbf))


def set_funnel(run):
    g = gains(run)
    C.FUN = dict(b=float(g['cbf_b']), x0=float(g['cbf_x0']), aL=float(g['cbf_aL']), y0L=float(g['cbf_y0L']),
                 aR=float(g['cbf_aR']), y0R=float(g['cbf_y0R']))
    return g


def label(run):
    if run.startswith('proposed'):
        return 'Proposed'
    return r'RaCBF, $\alpha_{dock}$ = ' + f"{alpha(run):g}"


def alpha(run):
    return float(gains(run)['alpha_dock'])


_cache = {}


def load(run):
    """(results list, {trial: verdict key}) with trajectories converted for the drawing helpers."""
    if run in _cache:
        return _cache[run]
    res = pickle.load(open(FF / 'runs' / f'{run}.pkl', 'rb'))['results']
    for r in res:
        t = r['traj']
        d = {k: np.asarray(v, float) for k, v in t.items()}
        d['xr'] = d['x'] - d['CD_x']; d['yr'] = d['y']
        d['xh'] = d['xr'] + C.HP * np.cos(d['psi']); d['yh'] = d['yr'] + C.HP * np.sin(d['psi'])
        r['traj'] = d
    ver = {}
    for row in csv.DictReader(open(FF / 'outcomes.csv')):
        if row['run'] == run:
            v = row['verdict']
            # The author's rule: a contact after the head point got into the pocket is an arrival,
            # not a collision, so 'pocket' is counted with the docks (2026-10-11).
            if v == 'pocket':
                v = 'success'
            if v == 'success' and int(row['qp_fail']) > 0:
                v = 'success_qp'
            ver[int(row['trial'])] = v
    for i, r in enumerate(res):
        r['clear_before_entry'] = clearance_before_entry(r, ver[i])
    _cache[run] = (res, ver)
    return res, ver


def clearance_before_entry(r, verdict):
    """Minimum clearance up to the first step the head point is inside the pocket.

    Collisions outside and timeouts never enter, so their plant level minimum is used as is.
    For docks the logged clearance is cut at the entry, since contact inside the pocket is an
    arrival by the agreed rule.
    """
    if verdict in ('outside', 'timeout'):
        return r['min_avoid']
    d = r['traj']
    inside = np.flatnonzero((d['xh'] >= -C.CX) & (np.abs(d['yh']) <= C.CY))
    end = inside[0] + 1 if len(inside) else len(d['avoid'])
    return float(np.min(d['avoid'][:end]))
