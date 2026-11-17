import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

os.makedirs('figures', exist_ok=True)

df = pd.read_csv('m5_results.csv').dropna(subset=['Model'])

models_mase = ['Croston', 'SBA', 'Plain Ens.', 'Hurdle-only', 'H-LGBM-TAN (mixture)']
mase_vals = [df.loc[df['Model']==m, 'MASE (mean)'].values[0] for m in models_mase]

models_spl = ['Hurdle-LGBM-QR', 'H-LGBM-TAN (hard)', 'H-LGBM-TAN Hybrid + CQR', 'H-LGBM-TAN (mixture)']
spl_vals = [df.loc[df['Model']==m, 'SPL (mean)'].values[0] for m in models_spl]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: MASE
colors1 = ['#B0BEC5', '#90A4AE', '#78909C', '#607D8B', '#E65100']
bars1 = ax1.bar(models_mase, mase_vals, color=colors1, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('MASE (Lower is Better)', fontweight='bold')
ax1.set_title('(a) Point Forecast Accuracy', fontweight='bold', pad=10)
ax1.set_xticks(np.arange(len(models_mase)))
ax1.set_xticklabels(['Croston', 'SBA', 'Plain Ens.', 'Hurdle-only', 'Proposed\n(Mixture)'], rotation=25, ha='right')

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=10)

# Subplot 2: SPL Explosion
colors2 = ['#90A4AE', '#607D8B', '#D32F2F', '#E65100']
bars2 = ax2.bar(models_spl, spl_vals, color=colors2, edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Scaled Pinball Loss (SPL)', fontweight='bold')
ax2.set_title('(b) The Conformal Explosion', fontweight='bold', pad=10)
ax2.set_xticks(np.arange(len(models_spl)))
ax2.set_xticklabels(['QR Baseline', 'Hard Coupling', 'Hybrid + CQR\n(Explosion)', 'Proposed\n(Mixture)'], rotation=25, ha='right')

# Use log scale because Hybrid+CQR explodes to 48 while others are ~0.3
ax2.set_yscale('log') 

for idx, bar in enumerate(bars2):
    yval = bar.get_height()
    weight = 'bold' if idx == 2 else 'normal'
    ax2.text(bar.get_x() + bar.get_width()/2, yval * 1.15, f'{yval:.3f}', ha='center', va='bottom', fontsize=10, fontweight=weight)

plt.suptitle('Figure 6: Scalability and Stability on M5 Walmart Dataset', fontsize=15, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('figures/fig6_m5_scalability_colored.png')
plt.savefig('figures/fig6_m5_scalability_colored.pdf')
print("Figure 6 generated.")
