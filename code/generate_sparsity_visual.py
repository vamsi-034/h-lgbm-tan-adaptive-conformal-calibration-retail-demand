import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
df = pd.read_csv('data/uci_online_retail_processed.csv')
sales = df['sales'].values

# Calculate stats
zero_pct = (sales == 0).mean() * 100
non_zeros = sales[sales > 0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left Plot: Zero vs Non-Zero
colors = ['#EF5350', '#29B6F6']
# FIX: removed the hardcoded 71.75% from the label so it doesn't double-print with autopct
ax1.pie([zero_pct, 100 - zero_pct], labels=['Zero Demand', 'Active Demand'], 
        autopct='%1.2f%%', startangle=90, colors=colors, 
        wedgeprops={'edgecolor': 'black', 'linewidth': 1})
ax1.set_title('Extreme Sparsity (Zero-Inflation)', fontweight='bold')

# Right Plot: Distribution of Non-Zero Demand (Lumpiness)
sns.histplot(non_zeros, bins=50, kde=False, color='#29B6F6', edgecolor='black', ax=ax2)
ax2.set_yscale('log')
ax2.set_xlabel('Order Size (Units)', fontweight='bold')
ax2.set_ylabel('Frequency (Log Scale)', fontweight='bold')
ax2.set_title('High Volatility in Active Demand', fontweight='bold')

# Add annotation about CV2
cv2 = (non_zeros.std() / non_zeros.mean()) ** 2
ax2.annotate(f'Highly "Lumpy"\n$CV^2$ = {cv2:.1f}', 
             xy=(0.5, 0.7), xycoords='axes fraction', 
             fontsize=12, fontweight='bold', color='darkred',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

plt.tight_layout()
out_path = 'figures/fig7_data_sparsity.jpg'
plt.savefig(out_path)
print("Fixed sparsity visual generated successfully!")
