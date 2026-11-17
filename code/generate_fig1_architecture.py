import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Segoe UI']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

os.makedirs('figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.axis('off')

# Corporate Palette (Same as Fig 2)
COL_DATA   = '#EAECEE'
COL_PREP   = '#D4E6F1'
COL_MODEL  = '#A9CCE3'
COL_COUP   = '#F9E79F'
COL_CAL    = '#E5E7E9'
COL_METRIC = '#D1F2EB'
EDGE_COLOR = '#1B2631'
TEXT_COLOR = '#17202A'

def draw_box(ax, x, y, width, height, text, facecolor, edgecolor, fontsize=10):
    shadow = FancyBboxPatch((x + 0.04, y - 0.04), width, height, boxstyle="square,pad=0.03", fc='#000000', alpha=0.1, lw=0, zorder=1)
    ax.add_patch(shadow)
    box = FancyBboxPatch((x, y), width, height, boxstyle="square,pad=0.03", fc=facecolor, ec=edgecolor, lw=1.5, zorder=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=TEXT_COLOR, zorder=3, linespacing=1.4)
    return (x, y, width, height)

def draw_arrow(ax, xA, yA, xB, yB, label=None, color=EDGE_COLOR, ls='-', label_color='#900C3F'):
    arrow = FancyArrowPatch((xA, yA), (xB, yB), arrowstyle='-|>', mutation_scale=15, lw=1.5, color=color, linestyle=ls, zorder=1)
    ax.add_patch(arrow)
    if label:
        mx, my = (xA + xB) / 2, (yA + yB) / 2
        ax.text(mx, my, label, ha='center', va='center', fontsize=9, color=label_color, fontweight='bold', zorder=4,
                bbox=dict(boxstyle='square,pad=0.2', fc='#FFFFFF', ec=label_color, lw=1.2, alpha=1.0))

# ROW 1 (y=7.5)
draw_box(ax, 3.0, 7.5, 6.0, 1.0, "Raw Time Series & Engineered Features", COL_DATA, EDGE_COLOR, 11)

# ROW 2 (y=5.8)
draw_box(ax, 0.5, 5.8, 3.3, 1.2, "STAGE 2: Distribution\nTemporal Attention Net\n(BiGRU + Multi-Head)\n$\\hat{q}_{10}, \\hat{q}_{50}, \\hat{q}_{90}$", COL_MODEL, EDGE_COLOR)
draw_box(ax, 4.35, 5.8, 3.3, 1.2, "STAGE 1: Occurrence Gate\n(LightGBM)\n$\\hat{p} = P(y > 0 | x)$", COL_MODEL, EDGE_COLOR)
draw_box(ax, 8.2, 5.8, 3.3, 1.2, "Point Ensemble\n(LightGBM / XGBoost)\n$\\hat{y}_{mean}$", COL_MODEL, EDGE_COLOR)

# ROW 3 (y=3.8)
draw_box(ax, 0.5, 3.8, 3.3, 1.2, "Static CQR\n(Conformal Shift)", COL_CAL, EDGE_COLOR)
draw_box(ax, 4.35, 3.8, 3.3, 1.2, "STAGE 3: Coupling Rules\n(Hard, Soft, Hybrid, Mixture)\n$g=\\mathbb{1}[\\hat{p}\\geq\\theta]$", COL_COUP, EDGE_COLOR)
draw_box(ax, 8.2, 3.8, 3.3, 1.2, "FINAL OUTPUT\nPoint Forecast\n$\\hat{y}$", COL_METRIC, EDGE_COLOR)

# ROW 4 (y=1.8)
draw_box(ax, 0.5, 1.8, 3.3, 1.2, "Adaptive Calibration\n(ACI / AgACI)", COL_CAL, EDGE_COLOR)
draw_box(ax, 4.35, 1.8, 3.3, 1.2, "FINAL OUTPUT\nCalibrated 80% Interval\n$\\hat{q}_{10}', \\hat{q}_{90}'$", COL_METRIC, EDGE_COLOR)


# ARROWS
# Input to Row 2
draw_arrow(ax, 6.0, 7.5, 6.0, 7.0) # To Gate
draw_arrow(ax, 4.0, 7.5, 2.15, 7.0) # To TAN
draw_arrow(ax, 8.0, 7.5, 9.85, 7.0) # To Pens

# Row 2 to Coupling
draw_arrow(ax, 6.0, 5.8, 6.0, 5.0) # Gate to Coupling (Straight down)
draw_arrow(ax, 2.15, 5.8, 4.35, 4.8) # TAN to Coupling (Diag down-right)
draw_arrow(ax, 9.85, 5.8, 7.65, 4.8) # Pens to Coupling (Diag down-left)

# Coupling Out
draw_arrow(ax, 4.35, 4.4, 3.8, 4.4) # Coupling to CQR (Straight left)
draw_arrow(ax, 7.65, 4.4, 8.2, 4.4) # Coupling to Pt Out (Straight right)
draw_arrow(ax, 6.0, 3.8, 6.0, 3.0, label='Mixture Bypasses\nCalibration', color='#900C3F', ls='--') # Mixture Bypass

# Calibration Chain
draw_arrow(ax, 2.15, 3.8, 2.15, 3.0) # CQR to ACI (Straight down)
draw_arrow(ax, 3.8, 2.4, 4.35, 2.4) # ACI to Int Out (Straight right)

# Grouping Box for Proposed Method
prop_box = FancyBboxPatch((0.2, 1.5), 7.7, 5.8, boxstyle="square,pad=0.02", fc='none', ec='#2C3E50', lw=1.5, linestyle='--', zorder=0)
ax.add_patch(prop_box)
ax.text(4.05, 7.15, "H-LGBM-TAN Architecture", ha='center', va='center', fontsize=11, fontweight='bold', color='#2C3E50')

plt.title('H-LGBM-TAN v4: Model Architecture', fontsize=16, fontweight='bold', color='#1B2631', pad=20)

plt.tight_layout()
plt.savefig("figures/fig1_architecture.png", bbox_inches="tight", pad_inches=0.1)
plt.savefig("figures/fig1_architecture.pdf", bbox_inches="tight", pad_inches=0.1)
plt.close()
print("Figure 1 matched to Figure 2 layout generated.")
