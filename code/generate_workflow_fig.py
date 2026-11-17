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

# Corporate Palette
COL_DATA   = '#EAECEE'
COL_PREP   = '#D4E6F1'
COL_MODEL  = '#A9CCE3'
COL_COUP   = '#F9E79F'
COL_CAL    = '#E5E7E9'
COL_METRIC = '#D1F2EB'
EDGE_COLOR = '#1B2631'
TEXT_COLOR = '#17202A'

def draw_box(ax, x, y, width, height, text, facecolor, edgecolor, fontsize=9.5):
    shadow = FancyBboxPatch((x + 0.04, y - 0.04), width, height, boxstyle="square,pad=0.03", fc='#000000', alpha=0.1, lw=0, zorder=1)
    ax.add_patch(shadow)
    box = FancyBboxPatch((x, y), width, height, boxstyle="square,pad=0.03", fc=facecolor, ec=edgecolor, lw=1.5, zorder=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=TEXT_COLOR, zorder=3, linespacing=1.4)

def draw_arrow(ax, xA, yA, xB, yB):
    arrow = FancyArrowPatch((xA, yA), (xB, yB), arrowstyle='-|>', mutation_scale=15, lw=1.5, color=EDGE_COLOR, zorder=1)
    ax.add_patch(arrow)

draw_box(ax, 0.5, 7.5, 4, 1, "Raw Datasets\n(UCI Retail, Car Parts, M5)", COL_DATA, EDGE_COLOR)
draw_box(ax, 5.5, 7.5, 6, 1, "Data Preprocessing & Filtering\n(Retain items with >50% zeros)", COL_PREP, EDGE_COLOR)
draw_arrow(ax, 4.5, 8.0, 5.5, 8.0)

draw_box(ax, 5.5, 6.0, 6, 1, "Rolling-Origin Evaluation Protocol\n(Time-based split, multiple folds)", COL_PREP, EDGE_COLOR)
draw_arrow(ax, 8.5, 7.5, 8.5, 7.0)

draw_box(ax, 0.5, 4.0, 3, 1, "Statistical Baselines\n(Croston, SBA, TSB)", COL_MODEL, EDGE_COLOR)
draw_box(ax, 4.0, 4.0, 3, 1, "Deep Learning\n(DeepAR, TFT)", COL_MODEL, EDGE_COLOR)
draw_box(ax, 7.5, 4.0, 4, 1, "H-LGBM-TAN (Proposed)\nStage 1 (Occurrence) + Stage 2 (Quantiles)", COL_MODEL, EDGE_COLOR)

draw_arrow(ax, 8.5, 6.0, 2.0, 5.0)
draw_arrow(ax, 8.5, 6.0, 5.5, 5.0)
draw_arrow(ax, 8.5, 6.0, 9.5, 5.0)

draw_box(ax, 7.5, 2.5, 4, 1, "Stage 3: Coupling Strategies\n(Hard, Soft, Hybrid, Mixture)", COL_COUP, EDGE_COLOR)
draw_arrow(ax, 9.5, 4.0, 9.5, 3.5)

draw_box(ax, 7.5, 1.0, 4, 1, "Conformal Calibration\n(Static CQR & Adaptive ACI)", COL_CAL, EDGE_COLOR)
draw_arrow(ax, 9.5, 2.5, 9.5, 2.0)

draw_box(ax, 0.5, 1.0, 6, 1, "Deterministic & Probabilistic Evaluation\n(MASE, RMSSE, Cov(all), Cov(occ))", COL_METRIC, EDGE_COLOR)

draw_arrow(ax, 2.0, 4.0, 2.0, 2.0)
draw_arrow(ax, 5.5, 4.0, 3.5, 2.0)
draw_arrow(ax, 7.5, 1.5, 6.5, 1.5)

prop_box = FancyBboxPatch((7.2, 0.7), 4.6, 4.6, boxstyle="square,pad=0.02", fc='none', ec='#2C3E50', lw=1.5, linestyle='--', zorder=0)
ax.add_patch(prop_box)
ax.text(9.5, 5.5, "Proposed Architecture Evaluation", ha='center', va='center', fontsize=11, fontweight='bold', color='#2C3E50')

plt.text(6.0, 8.8, 'Experimental Workflow Diagram', ha='center', fontsize=16, fontweight='bold', color='#1B2631')

plt.tight_layout()
plt.savefig("figures/fig2_workflow.png", bbox_inches="tight", pad_inches=0.1)
plt.savefig("figures/fig2_workflow.pdf", bbox_inches="tight", pad_inches=0.1)
plt.close()
print("Corporate Figure 2 generated.")
