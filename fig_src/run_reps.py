"""Representative recovery candidates (Fig. 6) on several seas, plus a stacked gallery."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cand_figs as K
import ff_common as F

SEAS = [28, 12, 30, 48, 56, 22]
for s in SEAS:
    K.fig6('proposed_as0.60', [F.RUNS['A'][0], F.RUNS['A'][1]], sea=s)
imgs = [mpimg.imread(F.OUTDIR / f'cand_fig6_rep_proposed_as0.60_sea{s}.png') for s in SEAS]
w = max(i.shape[1] for i in imgs); H = sum(i.shape[0] for i in imgs)
fig = plt.figure(figsize=(w / 150, H / 150))
y = 1.0
for im in imgs:
    h = im.shape[0] / H; y -= h
    fig.add_axes([0, y, im.shape[1] / w, h]).imshow(im)
for a in fig.axes:
    a.axis('off')
fig.savefig(F.OUTDIR / 'rep_gallery.png', dpi=150)
print('wrote', F.OUTDIR / 'rep_gallery.png')
