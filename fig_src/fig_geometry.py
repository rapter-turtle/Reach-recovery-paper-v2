"""Dimensioned schematic of the docking target and the collision set, at model scale."""
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc

import common as C

C.style()
XL, YL = (-2.35, 1.05), (-0.95, 1.25)
fig, ax = plt.subplots(figsize=(3.5, 2.45))
C.draw_geometry(ax, XL, YL, wall=True, target=True)
ax.set_xlabel(r'$x_r$ [m]')
ax.set_ylabel(r'$y_r$ [m]')


def dim(p0, p1, text, tp, color=C.INK, fs=8.5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='<->', mutation_scale=7, lw=0.7,
                                 color=color, zorder=9, shrinkA=0, shrinkB=0))
    ax.text(*tp, text, fontsize=fs, color=color, ha='center', va='center', zorder=10)


# USV pose, bow at the head point
psi = math.radians(-20.0)
cg = np.array([-1.62, 0.78])
ez = np.array([math.cos(psi), math.sin(psi)])
nz = np.array([-math.sin(psi), math.cos(psi)])
hp = cg + C.HP * ez
C.hull(ax, cg[0], cg[1], psi, fc='#cfe0f1', ec='#3b75af', alpha=1.0, lw=0.8)
ax.plot(*cg, 'o', ms=3.0, color=C.INK, zorder=10)
ax.plot(*hp, 'o', ms=3.4, color='#3b75af', zorder=10)
ax.plot([cg[0], hp[0]], [cg[1], hp[1]], color='#3b75af', lw=0.7, zorder=9)
ax.text(cg[0] - 0.36, cg[1] + 0.2, 'CoG', fontsize=8, ha='center', color=C.INK, zorder=10)
ax.plot([cg[0] - 0.27, cg[0] - 0.02], [cg[1] + 0.17, cg[1] + 0.02], color=C.INK, lw=0.5, zorder=10)
lm = 0.5 * (cg + hp) + 0.2 * nz
ax.text(*lm, r'$l_h$', fontsize=8.5, color='#3b75af', ha='center', va='center', zorder=10)
ax.plot([cg[0], cg[0] + 0.66], [cg[1], cg[1]], color='#888888', lw=0.6, ls='--', zorder=9)
ax.add_patch(Arc(cg, 1.0, 1.0, theta1=math.degrees(psi), theta2=0, lw=0.7, color=C.INK, zorder=9))
ax.text(cg[0] + 0.6, cg[1] - 0.09, r'$\psi_r$', fontsize=8.5, va='center', zorder=10)
ax.text(hp[0] - 0.02, hp[1] - 0.2, 'head point', fontsize=8, ha='center', color='#3b75af', zorder=10)

# pocket and arm dimensions
dim((0.0, -0.1), (C.CX, -0.1), r'$C_X$', (0.25, -0.03))
ax.plot([C.CX, C.CX], [-C.CY, C.CY], color='#999999', lw=0.5, ls=':', zorder=8)
dim((0.8, 0.0), (0.8, C.CY), r'$C_Y$', (0.92, 0.1))
ax.plot([0.0, 0.84], [0.0, 0.0], color='#999999', lw=0.5, ls=':', zorder=8)
ax.plot([C.CX + C.TW, 0.84], [C.CY, C.CY], color='#999999', lw=0.5, ls=':', zorder=8)
dim((-0.62, C.CY), (-0.62, C.CY + C.TW), r'$T_W$', (-0.76, C.CY + 0.07))
ax.plot([-0.66, -C.CX], [C.CY, C.CY], color='#999999', lw=0.5, ls=':', zorder=8)
ax.plot([-0.66, -C.CX], [C.CY + C.TW, C.CY + C.TW], color='#999999', lw=0.5, ls=':', zorder=8)
ax.plot(0, 0, '+', color=C.INK, ms=6, mew=0.9, zorder=10)
ax.text(-0.3, 0.06, r'$\mathcal{T}$', fontsize=9.5, color=C.SET_EDGE, ha='center', va='center', zorder=10)

# mothership side and approach wall
dim((-2.05, 0.0), (-2.05, C.Y_MS), r'$y_{MS}$', (-1.88, -0.27))
ax.plot([-2.1, -C.CX], [0.0, 0.0], color='#999999', lw=0.5, ls=':', zorder=1.8)
ax.text(-2.28, -0.78, 'mothership', fontsize=8, color=C.MS_EDGE, ha='left', va='center', zorder=10)
ax.text(0.25, 0.98, 'approach wall', fontsize=7.5, color=C.INK2, ha='center', va='center', zorder=10,
        bbox=dict(fc='white', ec='none', pad=0.6))
ax.add_patch(FancyArrowPatch((-0.2, -0.78), (0.6, -0.78), arrowstyle='-|>', mutation_scale=9,
                             lw=1.1, color='#1f2d4a', zorder=10))
ax.text(0.68, -0.78, r'$V_s$', fontsize=9, va='center', color='#1f2d4a', zorder=10)
C.save(fig, 'set_geometry')
