"""Screen all 100 proposed runs for the cleanest looking hand over set (Fig. 6 candidates).

    python screen_reps.py   -> prints the ranking, writes preview/candidates/rep_screen.csv

Fast pass on a 1 cm grid with the Z1 outline parameters scaled to that grid (cell counts halved).
Shape score: one piece connected to the pocket, high solidity (area over convex hull area), a
moderate size, the hand over head point inside, and clear of the approach wall.
"""
import csv

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

import ff_common as F
import common as C
import cand_figs as K

H = 0.01
XL, YL = (-3.0, 0.9), (-0.6, 2.0)


def mask_at(state):
    xs, ys, V, av, tg = C.value_slice(state, XL, YL, h=H)
    vs = V.copy(); vs[av] = 0.05; vs = ndimage.gaussian_filter(vs, 2.0); vs[av] = 1.0
    r = int(round(C.BAND_OFF / H)); yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    grown = ndimage.binary_dilation(vs <= C.BAND_DEEP, structure=(xx ** 2 + yy ** 2) <= r * r)
    _, v = C.author_polys(xs, ys, V, av, K.SET_LEVEL, smooth=2.0, openxy=(55, 3), rounding=5, maskblur=7,
                          allow=grown)
    return xs, ys, v <= K.SET_LEVEL, av


def score(xs, ys, m, av, head):
    X, Y = np.meshgrid(xs, ys)
    lab, n = ndimage.label(m)
    if n == 0:
        return dict(area=0, solidity=0, pieces=0, pocket=False, head_in=False, wall_gap=0, score=-1)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    main = lab == (1 + int(np.argmax(sizes)))
    area = main.sum() * H * H
    pts = np.column_stack([X[main], Y[main]])
    solidity = area / ConvexHull(pts).volume if len(pts) > 10 else 0.0
    pocket = bool((main & (np.abs(X) <= C.CX) & (np.abs(Y) <= C.CY)).any())
    ih = (int(np.argmin(np.abs(ys - head[1]))), int(np.argmin(np.abs(xs - head[0]))))
    head_in = bool(main[ih])
    wall = av & (X >= -C.CX) & (Y >= C.Y_WALL)
    d = ndimage.distance_transform_edt(~wall) * H
    wall_gap = float(d[main].min())
    s = solidity
    s -= 0.15 * (n - 1)                                 # extra islands look untidy
    s -= 0.0 if pocket else 0.3
    s -= 0.0 if head_in else 0.2
    s -= 0.1 if wall_gap < 0.05 else 0.0
    s -= 0.1 if not (0.5 <= area <= 2.2) else 0.0
    return dict(area=area, solidity=solidity, pieces=n, pocket=pocket, head_in=head_in, wall_gap=wall_gap, score=s)


if __name__ == '__main__':
    res, _ = F.load('proposed_as0.60')
    _, vc = F.load(F.RUNS['A'][0]); _, vt = F.load(F.RUNS['A'][1])
    rows = []
    for s, r in enumerate(res):
        d = r['traj']; ih = int(np.argmax(d['mode'] > 0.5)); k = ih - 1
        st = (d['psi'][k], d['u'][k], d['v'][k], d['r'][k], d['delta'][k], d['F'][k])
        xs, ys, m, av = mask_at(st)
        sc = score(xs, ys, m, av, (d['xh'][k], d['yh'][k]))
        rows.append(dict(sea=s, start_x=round(float(d['xr'][0]), 2), start_y=round(float(d['yr'][0]), 2),
                         r1=vc[s], r2=vt[s], **{kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in sc.items()}))
        print(s, rows[-1]['score'], flush=True)
    rows.sort(key=lambda x: -x['score'])
    out = F.OUTDIR / 'rep_screen.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('top 12:')
    for x in rows[:12]:
        print(x)
