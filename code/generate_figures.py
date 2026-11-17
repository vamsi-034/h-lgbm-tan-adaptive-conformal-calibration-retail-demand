import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set publication style
plt.style.use('default')
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Load data
df = pd.read_csv('results_summary.csv')
# drop empty lines if any
df = df.dropna(subset=['Model'])

# Create a figures directory
os.makedirs('figures', exist_ok=True)

# --- Figure 1: Point Forecast Accuracy (MASE) ---
# Match Table 1
mase_models = ['Croston', 'SBA', 'Plain Ens.', 'Hurdle-only', 'H-LGBM-TAN (hard)', 'H-LGBM-TAN (hybrid)']
mase_df = df[df['Model'].isin(mase_models)].copy()
# Ensure correct order
mase_df['Model'] = pd.Categorical(mase_df['Model'], categories=mase_models, ordered=True)
mase_df = mase_df.sort_values('Model')

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#B0BEC5', '#90A4AE', '#607D8B', '#455A64', '#FF9800', '#E65100']
bars = ax.bar(mase_df['Model'].astype(str), mase_df['MASE (mean)'], 
              yerr=mase_df['MASE (std)'], capsize=5, color=colors, edgecolor='black', linewidth=1)

ax.set_ylabel('MASE (Lower is better)', fontweight='bold')
ax.set_title('Point Forecast Accuracy on UCI Retail Dataset', fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval - 0.08, f'{yval:.3f}', 
            ha='center', va='top', color='white', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig1_mase_comparison.png')
plt.close()


# --- Figure 2: Probabilistic Calibration (Occurrence Coverage) ---
# Match the new trade-off narrative in Table 2
prob_models = ['H-LGBM-TAN (hard)', 'H-LGBM-TAN (mixture)', 'H-LGBM-TAN Hybrid + CQR']
prob_df = df[df['Model'].isin(prob_models)].copy()

name_map = {
    'H-LGBM-TAN (hard)': 'H-LGBM-TAN\n(Hard Gate)',
    'H-LGBM-TAN (mixture)': 'H-LGBM-TAN\n(Mixture)',
    'H-LGBM-TAN Hybrid + CQR': 'H-LGBM-TAN\nHybrid + CQR',
}
prob_df['Model_Plot'] = prob_df['Model'].map(name_map)
ordered_names = [name_map[m] for m in prob_models]
prob_df['Model_Plot'] = pd.Categorical(prob_df['Model_Plot'], categories=ordered_names, ordered=True)
prob_df = prob_df.sort_values('Model_Plot')

fig, ax = plt.subplots(figsize=(7, 5))
colors_prob = ['#CFD8DC', '#90CAF9', '#1565C0']

bars = ax.bar(prob_df['Model_Plot'].astype(str), prob_df['Cov(occ) (mean)'], 
              yerr=prob_df['Cov(occ) (std)'], capsize=5, color=colors_prob, edgecolor='black', linewidth=1)

# Target line
target_cov = 0.80
ax.axhline(target_cov, color='red', linestyle='--', linewidth=2, label=f'Target Coverage ({target_cov})')

ax.set_ylabel('Occurrence Coverage (Closer to target is better)', fontweight='bold')
ax.set_title('Uncertainty Calibration on Intermittent Demand', fontweight='bold', pad=15)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper left')

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval/2, f'{yval:.3f}', 
            ha='center', va='center', color='white', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('figures/fig2_probabilistic_coverage.png')
plt.close()


# --- Figure 3: Ablation Study - Coupling Methods ---
ablation_models = ['H-LGBM-TAN (hard)', 'H-LGBM-TAN (soft)', 'H-LGBM-TAN (hybrid)']
ab_df = df[df['Model'].isin(ablation_models)].copy()

fig, ax = plt.subplots(figsize=(6, 5))
colors_ab = ['#EF9A9A', '#EF5350', '#C62828']
names = ['Hard Coupling', 'Soft Coupling', 'Hybrid Coupling\n(Proposed)']

bars = ax.bar(names, ab_df['MASE (mean)'], 
              yerr=ab_df['MASE (std)'], capsize=5, color=colors_ab, edgecolor='black', linewidth=1)

ax.set_ylabel('MASE (Lower is better)', fontweight='bold')
ax.set_title('Ablation Study: Coupling Methods', fontweight='bold', pad=15)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval - 0.05, f'{yval:.3f}', 
            ha='center', va='top', color='white', fontweight='bold', fontsize=12)

# Highlight Hybrid/Soft Tie
ax.text(2, 0.65, 'Mathematically tied\nwith Soft Coupling', 
        ha='center', va='bottom', color='darkred', fontweight='bold', style='italic')

plt.tight_layout()
plt.savefig('figures/fig3_ablation_coupling.png')
plt.close()

print("Figures generated successfully in 'figures' directory!")
