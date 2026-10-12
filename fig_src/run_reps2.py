"""Fig. 6 candidates on the best screened seas (screen_reps.py), plus a two column gallery."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cand_figs as K
import ff_common as F

SEAS = [29, 96, 71, 63, 19, 34, 94, 38]
for s in SEAS:
    K.fig6('proposed_as0.60', [F.RUNS['A'][0], F.RUNS['A'][1]], sea=s)
    print('done sea', s, flush=True)
imgs = [mpimg.imread(F.OUTDIR / f'cand_fig6_rep_proposed_as0.60_sea{s}.png') for s in SEAS]
rows = [imgs[i:i + 2] for i in range(0, len(imgs), 2)]
cw = max(i.shape[1] for i in imgs); rh = [max(i.shape[0] for i in r) for r in rows]
W, H = 2 * cw, sum(rh)
fig = plt.figure(figsize=(W / 150, H / 150))
y = 1.0
for r, h in zip(rows, rh):
    y -= h / H
    for c, im in enumerate(r):
        fig.add_axes([c * cw / W, y + (h - im.shape[0]) / H, im.shape[1] / W, im.shape[0] / H]).imshow(im)
for a in fig.axes:
    a.axis('off')
fig.savefig(F.OUTDIR / 'rep_gallery2.png', dpi=150)
print('wrote', F.OUTDIR / 'rep_gallery2.png', flush=True)
