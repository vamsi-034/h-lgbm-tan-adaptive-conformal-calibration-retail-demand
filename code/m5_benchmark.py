"""
Timing benchmark: run 10 random M5 series through 1 split of the full pipeline.
"""
import time, sys, os
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
os.chdir(r'C:\Users\asus\.gemini\antigravity-ide\scratch\h_lgbm_tan_v4')

N_BENCH = 10

print("="*60)
print(f"BENCHMARK: Full pipeline on {N_BENCH} random M5 series, 1 split")
print("="*60)

# Step 1: Load M5 data with random sampling
from pathlib import Path
t_start = time.time()

m5_path = Path("m5_data")
print("[1/4] Loading CSVs...")
sales = pd.read_csv(m5_path / "sales_train_evaluation.csv")
calendar = pd.read_csv(m5_path / "calendar.csv", parse_dates=["date"])
prices = pd.read_csv(m5_path / "sell_prices.csv")
t_csv = time.time() - t_start
print(f"  CSV load: {t_csv:.1f}s")

print("[2/4] Melting & filtering...")
id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
d_cols = [c for c in sales.columns if c.startswith("d_")]
long = sales.melt(id_vars=id_cols, value_vars=d_cols, var_name="d", value_name="sales")
long = long.merge(calendar[["d", "date", "event_name_1", "snap_CA"]], on="d", how="left")

item_stats = long.groupby("id").agg(
    n_days=("sales", "count"),
    n_nonzero=("sales", lambda x: (x > 0).sum()),
    total=("sales", "sum")
)
item_stats["adi"] = item_stats["n_days"] / item_stats["n_nonzero"].clip(lower=1)
intermittent = item_stats[item_stats["adi"] > 1.32].index
print(f"  {len(intermittent)} intermittent series found")

np.random.seed(42)
selected = np.random.choice(intermittent, size=N_BENCH, replace=False)
long = long[long["id"].isin(selected)].copy()

long["wm_yr_wk"] = long.merge(calendar[["d", "wm_yr_wk"]], on="d", how="left")["wm_yr_wk"]
long["item_store"] = long["item_id"] + "_" + long["store_id"]
prices["item_store"] = prices["item_id"] + "_" + prices["store_id"]
long = long.merge(prices[["item_store", "wm_yr_wk", "sell_price"]],
                  on=["item_store", "wm_yr_wk"], how="left")
long["sell_price"] = long["sell_price"].ffill().bfill().fillna(0)
long["series_id"] = long["id"]
long["price"] = long["sell_price"]
long["promo"] = long["event_name_1"].notna().astype(int)
long["day_of_week"] = long["date"].dt.dayofweek
long["month"] = long["date"].dt.month

df = long[["date", "series_id", "sales", "price", "promo",
           "day_of_week", "month"]].reset_index(drop=True)
df = df.sort_values(["series_id", "date"]).reset_index(drop=True)

t_prep = time.time() - t_start
print(f"  Data prep total: {t_prep:.1f}s")
print(f"  {len(df['series_id'].unique())} series, {len(df)} rows")

# Step 3: Build features & create 1 split
print("[3/4] Building features...")
from H_LGBM_TAN_v4 import build_features, rolling_origin_splits, run_single_split, CONFIG

processed_df = build_features(df, frequency="daily")
processed_df.dropna(inplace=True)
t_features = time.time() - t_start
print(f"  Feature build: {t_features - t_prep:.1f}s")

splits = rolling_origin_splits(processed_df, n_origins=1, 
                                test_horizon_days=CONFIG["test_horizon_days"],
                                min_train_frac=CONFIG["train_frac_initial"])
print(f"  1 split created")

# Step 4: Run the full pipeline on 1 split, 1 seed
print("[4/4] Running full pipeline (1 origin, 1 seed)...")
split = splits[0]
t_pipeline_start = time.time()

res = run_single_split(
    train_inner=split["train_inner"],
    val_df=split["val"],
    test_df=split["test"],
    train_full=split["train_full"],
    processed_df=processed_df,
    seed=42,
    test_horizon_days=CONFIG["test_horizon_days"],
    seq_len=28,
    verbose=False,
)
t_pipeline = time.time() - t_pipeline_start
t_total = time.time() - t_start

# Print results
print(f"\n{'='*60}")
print(f"TIMING RESULTS ({N_BENCH} series)")
print(f"{'='*60}")
print(f"  Data load + prep:    {t_prep:.0f}s")
print(f"  Feature build:       {t_features - t_prep:.0f}s")
print(f"  Pipeline (1 split):  {t_pipeline:.0f}s ({t_pipeline/60:.1f} min)")
print(f"  Total:               {t_total:.0f}s ({t_total/60:.1f} min)")
print()

# Full experiment = 3 origins x 5 seeds = 15 splits
full_pipeline_time = t_pipeline * 15  # 3 origins x 5 seeds
per_series_full = full_pipeline_time / N_BENCH

print(f"FULL EXPERIMENT EXTRAPOLATION (3 origins x 5 seeds = 15 splits):")
print(f"  Est. {N_BENCH}-series full run:  {full_pipeline_time:.0f}s ({full_pipeline_time/60:.1f} min)")
print(f"  Per-series cost (full):    {per_series_full:.1f}s ({per_series_full/60:.1f} min)")
print()

budget = 8 * 3600
overhead = t_prep + (t_features - t_prep) + 600  # data + features + safety
available = budget - overhead
max_series = int(available / (per_series_full / N_BENCH)) if per_series_full > 0 else 9999

print(f"8-HOUR BUDGET ({budget}s, overhead={overhead:.0f}s):")
print(f"  Available for pipeline:    {available:.0f}s")
print(f"  Max series (full CV):      ~{max_series}")
print(f"  Recommended (80% safety):  ~{int(max_series * 0.8)}")
print(f"{'='*60}")

# Also print some quick results to see if the model works at all
print(f"\nQUICK SANITY CHECK - Model results on {N_BENCH} M5 series:")
for model_name, metrics in res.items():
    if isinstance(metrics, dict) and 'MASE' in metrics:
        print(f"  {model_name}: MASE={metrics.get('MASE', 'N/A'):.3f}, SPL={metrics.get('SPL', 'N/A'):.3f}")
