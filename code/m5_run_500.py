"""
M5 Full Experiment: 500 randomly sampled intermittent series (~1 Million Rows)
Memory-optimized: filter series BEFORE melting to avoid memory explosion.
"""
import time
import sys
import os
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
os.chdir(r'C:\Users\asus\.gemini\antigravity-ide\scratch\h_lgbm_tan_v4')

N_SERIES = 500
RANDOM_SEED = 42

print("=" * 70)
print(f"  M5 FULL EXPERIMENT: {N_SERIES} randomly sampled intermittent series")
print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

import H_LGBM_TAN_v4 as pipeline

def load_m5_random_500(m5_dir):
    from pathlib import Path
    m5_path = Path(m5_dir)
    
    t0 = time.time()
    print(f"[M5-500] Loading M5 CSVs...")
    sales = pd.read_csv(m5_path / "sales_train_evaluation.csv")
    calendar = pd.read_csv(m5_path / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(m5_path / "sell_prices.csv")
    print(f"[M5-500] CSV load: {time.time()-t0:.1f}s")
    
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    
    print(f"[M5-500] Computing intermittency in wide format...")
    sales_vals = sales[d_cols].values
    n_days = sales_vals.shape[1]
    n_nonzero = (sales_vals > 0).sum(axis=1)
    totals = sales_vals.sum(axis=1)
    adi = n_days / np.clip(n_nonzero, 1, None)
    
    intermittent_mask = adi > 1.32
    print(f"[M5-500] Found {intermittent_mask.sum()} intermittent series out of {len(sales)}")
    
    np.random.seed(RANDOM_SEED)
    intermittent_indices = np.where(intermittent_mask)[0]
    n_select = min(N_SERIES, len(intermittent_indices))
    selected_indices = np.random.choice(intermittent_indices, size=n_select, replace=False)
    
    sales_filtered = sales.iloc[selected_indices].copy()
    print(f"[M5-500] Filtered to {len(sales_filtered)} series BEFORE melt")
    
    print(f"[M5-500] Melting {len(sales_filtered)} series...")
    long = sales_filtered.melt(id_vars=id_cols, value_vars=d_cols,
                                var_name="d", value_name="sales")
    print(f"[M5-500] Melted to {len(long):,} rows")
    
    long = long.merge(calendar[["d", "date", "event_name_1", "snap_CA"]],
                      on="d", how="left")
    
    zero_rate = (long["sales"] == 0).mean()
    print(f"[M5-500] Zero-demand rate: {zero_rate*100:.2f}%")
    
    print(f"[M5-500] Merging prices...")
    long["wm_yr_wk"] = long.merge(calendar[["d", "wm_yr_wk"]], on="d",
                                   how="left")["wm_yr_wk"]
    long["item_store"] = long["item_id"] + "_" + long["store_id"]
    prices["item_store"] = prices["item_id"] + "_" + prices["store_id"]
    long = long.merge(prices[["item_store", "wm_yr_wk", "sell_price"]],
                      on=["item_store", "wm_yr_wk"], how="left")
    long["sell_price"] = long["sell_price"].ffill().bfill().fillna(0)
    
    del sales, sales_filtered, sales_vals
    import gc; gc.collect()
    
    long["series_id"] = long["id"]
    long["price"] = long["sell_price"]
    long["promo"] = long["event_name_1"].notna().astype(int)
    long["day_of_week"] = long["date"].dt.dayofweek
    long["month"] = long["date"].dt.month
    
    df = long[["date", "series_id", "sales", "price", "promo",
               "day_of_week", "month"]].reset_index(drop=True)
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    
    del long; gc.collect()
    
    elapsed = time.time() - t0
    print(f"[M5-500] Loaded {len(df['series_id'].unique())} series, {len(df):,} rows in {elapsed:.1f}s")
    return df

pipeline.load_m5_intermittent = load_m5_random_500

print(f"\n[MAIN] Starting full experiment at {time.strftime('%H:%M:%S')}")
t_experiment_start = time.time()

try:
    pipeline.run_full_experiment(dataset="m5", m5_dir="m5_data")
except Exception as e:
    import traceback
    print(f"\n[ERROR] Experiment failed: {e}")
    traceback.print_exc()

t_experiment_total = time.time() - t_experiment_start
print(f"\n{'='*70}")
print(f"  EXPERIMENT COMPLETE")
print(f"  Total time: {t_experiment_total:.0f}s ({t_experiment_total/3600:.1f} hours)")
print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*70}")
