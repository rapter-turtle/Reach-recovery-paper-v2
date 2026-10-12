"""Hand over set of sea 94 (Fig. 6 candidate) with the bulge beside the cradle trimmed.

    python cand_rep94.py -> preview/candidates/rep94_set_variants.png

V0 is the current Z1 band.  The others additionally trim the region beside and above the cradle
(common.corner_weight) to a copy of a deeper contour moved outward, or use a narrower or deeper
band.  Every variant only removes area from the V <= -0.006 set.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy import ndimage

import ff_common as F
import common as C
import cand_figs as K

C.style()
SEA = 94
XL, YL = (-3.2, 1.0), (-0.6, 1.3)
res, _ = F.load('proposed_as0.60')
d = res[SEA]['traj']; ih = int(np.argmax(d['mode'] > 0.5)); k = ih - 1
st = (d['psi'][k], d['u'][k], d['v'][k], d['r'][k], d['delta'][k], d['F'][k])
xs, ys, V, av, tg = C.value_slice(st, XL, YL, h=C.AUTHOR_STEP)
vs = V.copy(); vs[av] = 0.05; vs = ndimage.gaussian_filter(vs, 4.0); vs[av] = 1.0
h = C.AUTHOR_STEP
w = C.corner_weight(xs, ys)


def grow(level, off):
    r = int(round(off / h)); yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return ndimage.binary_dilation(vs <= level, structure=(xx ** 2 + yy ** 2) <= r * r)


band = grow(C.BAND_DEEP, C.BAND_OFF)
X, Y = np.meshgrid(xs, ys)
sx = np.clip((X + 1.1) / 0.75, 0, 1); sx = sx * sx * (3 - 2 * sx)          # 0 at x = -1.1, 1 at the mouth
taper = Y <= 0.55 - (0.55 - C.CY - 0.02) * sx                               # cosmetic cap easing into the mouth
VARIANTS = [   # key, note, allow mask, shift
    ('V0', 'current: band 0.15 m around V = -0.010', band, None),
    ('S1', 'band + beside the wall level to V = -0.014 (0.30 m)', band, C.wall_shift(xs, ys, av, K.SET_LEVEL, -0.014, 0.30)),
    ('S2', 'band + beside the wall level to V = -0.016 (0.35 m)', band, C.wall_shift(xs, ys, av, K.SET_LEVEL, -0.016, 0.35)),
    ('S3', 'band + beside the wall level to V = -0.018 (0.40 m)', band, C.wall_shift(xs, ys, av, K.SET_LEVEL, -0.018, 0.40)),
    ('S4', 'band + beside the wall level to V = -0.020 (0.45 m)', band, C.wall_shift(xs, ys, av, K.SET_LEVEL, -0.020, 0.45)),
    ('T', 'band + cosmetic taper into the mouth (not a V level)', band & taper, None),
]
fig, axs = plt.subplots(2, 3, figsize=(8.4, 4.2), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.05, hspace=0.3))
for ax, (key, note, allow, shift) in zip(axs.ravel(), VARIANTS):
    C.draw_geometry(ax, XL, YL, wall=True, target=True)
    ax.contour(xs, ys, vs, levels=[-0.014, -0.012, -0.010], colors=['#2171b5'], linewidths=0.4, alpha=0.8, zorder=3.4)
    for p in C.author_polys(xs, ys, V, av, K.SET_LEVEL, allow=allow, shift=shift)[0]:
        ax.add_patch(Polygon(p, closed=True, fc=C.SET_FILL, ec='none', alpha=0.35, zorder=2.5))
        ax.plot(p[:, 0], p[:, 1], color=C.SET_LINE, lw=0.9, zorder=2.6)
    ax.plot(d['xr'][:ih], d['yr'][:ih], color=C.PH1, lw=1.0, zorder=7)
    ax.plot(d['xr'][ih - 1:], d['yr'][ih - 1:], color=C.PH2, lw=1.0, zorder=7)
    ax.plot(d['xh'][k], d['yh'][k], 'o', ms=3, color='#c2185b', zorder=9)
    ax.plot(d['xr'][k], d['yr'][k], 'o', ms=4.5, color='k', zorder=10)
    ax.plot(*C.DP, 'D', ms=3.5, color='#333333', mec='white', mew=0.4, zorder=10)
    ax.set_title(f'{key}  {note}', fontsize=5.8)
fig.suptitle(f'Hand over set of sea {SEA}: outline variants (blue: smoothed V = -0.010, -0.012, -0.014)',
             fontsize=8, y=0.99, color=C.INK2)
out = F.OUTDIR / 'rep94_set_variants.png'
fig.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.03)
print('wrote', out)
