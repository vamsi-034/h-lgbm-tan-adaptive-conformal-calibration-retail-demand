#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H-LGBM-TAN v4: Publication-Ready Dual-Stage Hurdle-LightGBM + Temporal Attention Network
=========================================================================================

Upgraded from v3 to address all critical reviewer concerns:
  1. Real-data integration (UCI Online Retail II, M5, or any CSV)
  2. Soft-coupled point forecast (fixes the 2.6x MASE degradation)
  3. Adaptive Conformal Inference (fixes the 18pp coverage gap)
  4. Rolling-origin cross-validation (fixes single-split weakness)
  5. Expanded metrics: CRPS, per-demand-class breakdown, timing

Usage:
    python H_LGBM_TAN_v4.py                          # Run on synthetic data
    python H_LGBM_TAN_v4.py --dataset uci             # Run on UCI Online Retail II
    python H_LGBM_TAN_v4.py --dataset csv --csv-path /path/to/data.csv
    python H_LGBM_TAN_v4.py --dataset m5 --m5-dir /path/to/m5/

Convert to notebook:  jupytext --to notebook H_LGBM_TAN_v4.py
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, f1_score
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# 0. CONFIGURATION
# ============================================================================

CONFIG = {
    "seeds": [42, 43, 44, 45, 46],          # 5 seeds (up from 3)
    "n_origins": 3,                           # rolling-origin folds
    "test_horizon_days": 90,                  # test window per origin
    "train_frac_initial": 0.60,               # minimum training fraction
    "stage2_max_epochs": 60,
    "stage2_patience": 6,
    "stage2_batch_size": 64,
    "stage2_lr": 1e-3,
    "target_coverage": 0.80,
    "aci_gamma": 0.02,                        # ACI step size

    "uci_n_series": 200,                      # (up from 50)
    "synthetic_n_series": 25,
    "synthetic_n_days": 750,
}


# ============================================================================
# 1. DATA GENERATION (SYNTHETIC â€” kept as controlled ablation)
# ============================================================================

def generate_intermittent_data(n_series: int = 25, n_days: int = 750,
                                seed: int = 42) -> pd.DataFrame:
    """Synthetic intermittent-demand generator with unobserved latent factor.
    This creates an irreducible error floor, mimicking real-world conditions
    where the model cannot perfectly recover the true DGP."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")
    all_data = []

    for s in range(n_series):
        base_rate = rng.uniform(5.0, 10.0)
        promo = (rng.random(n_days) < 0.08).astype(int)
        price = rng.uniform(15.0, 25.0, size=n_days)
        price = np.where(promo == 1, price * rng.uniform(0.3, 0.6), price)

        day_of_week = dates.dayofweek.values
        month = dates.month.values

        # Unobserved latent per-SKU affinity: slow random walk
        latent = np.zeros(n_days)
        for t in range(1, n_days):
            latent[t] = 0.98 * latent[t - 1] + rng.normal(0, 0.15)

        sales = np.zeros(n_days, dtype=float)
        time_since_last = 0

        for t in range(n_days):
            time_since_last += 1
            logit = (
                -4.0
                + 0.18 * time_since_last
                + 3.2 * promo[t]
                + 0.8 * (day_of_week[t] >= 5)
                - 0.09 * price[t]
                + 1.1 * latent[t]
                + rng.normal(0, 0.5)
            )
            prob = 1.0 / (1.0 + np.exp(-logit))

            if rng.random() < prob:
                magnitude = rng.poisson(
                    lam=max(0.5, base_rate * (2.6 if promo[t] else 1.0) * np.exp(0.15 * latent[t]))
                ) + 1
                sales[t] = magnitude
                time_since_last = 0

        df = pd.DataFrame({
            "date": dates,
            "series_id": f"SKU_{s:03d}",
            "sales": sales,
            "price": price,
            "promo": promo,
            "day_of_week": day_of_week,
            "month": month,
        })
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


# ============================================================================
# 2. REAL DATA LOADERS
# ============================================================================

def load_uci_online_retail(cache_dir: str = "./data") -> pd.DataFrame:
    """Download and preprocess UCI Online Retail II dataset.
    Aggregates to daily sales per StockCode. Filters to items with
    >= 200 days of history and > 50% zero-demand days."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    csv_cache = cache_path / "uci_online_retail_processed.csv"

    if csv_cache.exists():
        print(f"[UCI] Loading cached data from {csv_cache}")
        df = pd.read_csv(csv_cache, parse_dates=["date"])
        return df

    print("[UCI] Downloading UCI Online Retail II dataset...")
    url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
    try:
        import zipfile, io, urllib.request
        response = urllib.request.urlopen(url, timeout=60)
        with zipfile.ZipFile(io.BytesIO(response.read())) as zf:
            # Find the Excel file inside
            excel_files = [f for f in zf.namelist() if f.endswith('.xlsx')]
            if not excel_files:
                raise FileNotFoundError("No .xlsx found in ZIP")
            with zf.open(excel_files[0]) as ef:
                raw = pd.read_excel(ef, engine='openpyxl')
    except Exception as e:
        print(f"[UCI] Download failed: {e}")
        print("[UCI] Please download manually from:")
        print(f"      {url}")
        print("      Extract the .xlsx and convert to CSV, or use --dataset synthetic")
        raise

    # Clean and aggregate
    raw = raw.dropna(subset=["StockCode", "InvoiceDate", "Quantity"])
    raw = raw[raw["Quantity"] > 0]
    raw["date"] = pd.to_datetime(raw["InvoiceDate"]).dt.date
    raw["date"] = pd.to_datetime(raw["date"])

    daily = (raw.groupby(["StockCode", "date"])
             .agg(sales=("Quantity", "sum"), price=("Price", "mean"))
             .reset_index())

    # Create full date range per item and fill missing days with 0
    date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    all_items = daily["StockCode"].unique()

    full_data = []
    for item in all_items:
        item_df = daily[daily["StockCode"] == item].set_index("date")
        item_full = item_df.reindex(date_range)
        item_full["sales"] = item_full["sales"].fillna(0)
        item_full["price"] = item_full["price"].ffill().bfill().fillna(0)
        item_full["StockCode"] = item
        item_full["date"] = item_full.index
        full_data.append(item_full.reset_index(drop=True))

    df = pd.concat(full_data, ignore_index=True)

    # Filter: >= 200 days history, > 50% zero-demand
    item_stats = df.groupby("StockCode").agg(
        n_days=("sales", "count"),
        zero_frac=("sales", lambda x: (x == 0).mean())
    )
    valid_items = item_stats[(item_stats["n_days"] >= 200) &
                              (item_stats["zero_frac"] > 0.50)].index
    df = df[df["StockCode"].isin(valid_items)].copy()

    # Limit to top N items by total volume
    n_series = CONFIG.get("uci_n_series", 200)
    top_items = (df[df["sales"] > 0].groupby("StockCode")["sales"].sum()
                 .nlargest(n_series).index)
    df = df[df["StockCode"].isin(top_items)].copy()

    # Add required columns
    df["series_id"] = df["StockCode"].astype(str)
    df["promo"] = 0  # UCI dataset doesn't have explicit promo flags
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    df = df[["date", "series_id", "sales", "price", "promo",
             "day_of_week", "month"]].reset_index(drop=True)

    df.to_csv(csv_cache, index=False)
    print(f"[UCI] Processed {len(df['series_id'].unique())} items, "
          f"{len(df)} rows, cached to {csv_cache}")
    return df


def load_m5_intermittent(m5_dir: str) -> pd.DataFrame:
    """Load M5 competition data and filter to intermittent items.
    Expects m5_dir to contain: sales_train_evaluation.csv, calendar.csv,
    sell_prices.csv (standard M5 layout from Kaggle)."""
    m5_path = Path(m5_dir)
    sales_path = m5_path / "sales_train_evaluation.csv"
    calendar_path = m5_path / "calendar.csv"
    prices_path = m5_path / "sell_prices.csv"

    for p in [sales_path, calendar_path, prices_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"M5 file not found: {p}\n"
                f"Download from https://www.kaggle.com/c/m5-forecasting-accuracy/data"
            )

    print("[M5] Loading M5 data...")
    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path, parse_dates=["date"])
    prices = pd.read_csv(prices_path)

    # Melt sales to long format
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(id_vars=id_cols, value_vars=d_cols,
                      var_name="d", value_name="sales")

    # Merge with calendar for dates
    long = long.merge(calendar[["d", "date", "event_name_1", "snap_CA"]],
                      on="d", how="left")

    # Filter to intermittent items (ADI > 1.32)
    item_stats = long.groupby("id").agg(
        n_days=("sales", "count"),
        n_nonzero=("sales", lambda x: (x > 0).sum()),
        total=("sales", "sum")
    )
    item_stats["adi"] = item_stats["n_days"] / item_stats["n_nonzero"].clip(lower=1)
    intermittent = item_stats[item_stats["adi"] > 1.32].index

    # Take top 50 by total volume from intermittent items
    top_items = item_stats.loc[intermittent].nlargest(50, "total").index
    long = long[long["id"].isin(top_items)].copy()

    # Merge prices (approximate: weekly prices mapped to daily)
    long["wm_yr_wk"] = long.merge(calendar[["d", "wm_yr_wk"]], on="d",
                                    how="left")["wm_yr_wk"]
    long["item_store"] = long["item_id"] + "_" + long["store_id"]
    prices["item_store"] = prices["item_id"] + "_" + prices["store_id"]
    long = long.merge(prices[["item_store", "wm_yr_wk", "sell_price"]],
                      on=["item_store", "wm_yr_wk"], how="left")
    long["sell_price"] = long["sell_price"].ffill().bfill().fillna(0)

    # Map to standard schema
    long["series_id"] = long["id"]
    long["price"] = long["sell_price"]
    long["promo"] = long["event_name_1"].notna().astype(int)
    long["day_of_week"] = long["date"].dt.dayofweek
    long["month"] = long["date"].dt.month

    df = long[["date", "series_id", "sales", "price", "promo",
               "day_of_week", "month"]].reset_index(drop=True)
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)

    print(f"[M5] Loaded {len(df['series_id'].unique())} intermittent items, {len(df)} rows")
    return df

def load_carparts_csv(csv_path: str = "carparts.csv") -> pd.DataFrame:
    """Loads the carparts dataset and applies standard intermittent filtering."""
    df = pd.read_csv(csv_path)
    if not {"series_id", "date", "sales"}.issubset(df.columns):
        raise ValueError("carparts.csv must contain series_id, date, and sales columns.")
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for >50% zero-rate
    zero_rates = df.groupby('series_id')['sales'].apply(lambda x: (x == 0).mean())
    valid_ids = zero_rates[zero_rates > 0.50].index
    df = df[df['series_id'].isin(valid_ids)].copy()
    print(f"[load_carparts] Filtered to {len(valid_ids)} series with >50% zero rate.")
    return df

def load_generic_csv(csv_path: str) -> pd.DataFrame:
    """Load a user-supplied CSV with columns:
    date, series_id, sales, price, promo, day_of_week, month"""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    required = ["date", "series_id", "sales", "price", "promo", "day_of_week", "month"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def load_data(dataset: str = "synthetic", seed: int = 42, **kwargs) -> pd.DataFrame:
    """Unified data loader dispatching to the appropriate source."""
    if dataset == "synthetic":
        return generate_intermittent_data(
            n_series=CONFIG["synthetic_n_series"],
            n_days=CONFIG["synthetic_n_days"],
            seed=seed,
        )
    elif dataset == "uci":
        return load_uci_online_retail(cache_dir=kwargs.get("cache_dir", "./data"))
    elif dataset == "m5":
        return load_m5_intermittent(kwargs["m5_dir"])
    elif dataset == "carparts":
        return load_carparts_csv(kwargs.get("csv_path", "carparts.csv"))
    elif dataset == "csv":
        return load_generic_csv(kwargs["csv_path"])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ============================================================================
# 3. DEMAND CLASSIFICATION (Syntetos-Boylan)
# ============================================================================

def classify_demand_syntetos_boylan(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each series_id into demand categories:
    smooth, erratic, intermittent, lumpy.
    Returns a DataFrame with series_id, ADI, CV2, and category."""
    records = []
    for sid, g in df.groupby("series_id"):
        sales = g["sales"].values
        non_zero = sales[sales > 0]
        if len(non_zero) < 2:
            records.append({"series_id": sid, "ADI": np.inf, "CV2": np.nan,
                            "category": "intermittent"})
            continue
        adi = len(sales) / len(non_zero)
        cv2 = (non_zero.std() / non_zero.mean()) ** 2

        if adi < 1.32 and cv2 < 0.49:
            cat = "smooth"
        elif adi < 1.32:
            cat = "erratic"
        elif cv2 < 0.49:
            cat = "intermittent"
        else:
            cat = "lumpy"
        records.append({"series_id": sid, "ADI": adi, "CV2": cv2, "category": cat})

    return pd.DataFrame(records)


# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

def build_features(df: pd.DataFrame, frequency: str = "daily") -> pd.DataFrame:
    df = df.copy()
    # Sort rigorously
    df.sort_values(by=['series_id', 'date'], inplace=True)
    
    # 1. Target variables
    # (occ is removed from features to prevent target leakage, computed locally when needed)
    
    # 2. Basic temporal features
    if frequency == "daily":
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
        
        # 3. Autoregressive features (Daily)
        df['lag_7'] = df.groupby('series_id')['sales'].shift(7)
        df['lag_14'] = df.groupby('series_id')['sales'].shift(14)
        df['lag_28'] = df.groupby('series_id')['sales'].shift(28)
        
        df['roll_mean_7'] = df.groupby('series_id')['sales'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
        df['roll_mean_28'] = df.groupby('series_id')['sales'].transform(lambda x: x.shift(1).rolling(28, min_periods=1).mean())
    else:
        # Monthly features
        df['month_of_year'] = df['date'].dt.month
        df['month_of_year_sin'] = np.sin(2 * np.pi * df['month_of_year'] / 12.0)
        df['month_of_year_cos'] = np.cos(2 * np.pi * df['month_of_year'] / 12.0)
        
        # Autoregressive features (Monthly)
        df['lag_1'] = df.groupby('series_id')['sales'].shift(1)
        df['lag_3'] = df.groupby('series_id')['sales'].shift(3)
        df['lag_6'] = df.groupby('series_id')['sales'].shift(6)
        df['lag_12'] = df.groupby('series_id')['sales'].shift(12)
        
        df['roll_mean_3'] = df.groupby('series_id')['sales'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df['roll_mean_6'] = df.groupby('series_id')['sales'].transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
        
    return df.dropna().reset_index(drop=True)


ID_COLS = ["date", "series_id", "sales"]


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ID_COLS]


def time_split(df: pd.DataFrame, train_frac: float = 0.8):
    unique_dates = np.sort(df["date"].unique())
    split_idx = int(len(unique_dates) * train_frac)
    train_dates, test_dates = unique_dates[:split_idx], unique_dates[split_idx:]
    train_df = df[df["date"].isin(train_dates)].reset_index(drop=True)
    test_df = df[df["date"].isin(test_dates)].reset_index(drop=True)
    return train_df, test_df


def rolling_origin_splits(df: pd.DataFrame, n_origins: int = 3,
                           test_horizon_days: int = 90,
                           min_train_frac: float = 0.60):
    """Generate multiple train/test splits with expanding training window.
    Returns list of (train_df, val_df, test_df) tuples."""
    unique_dates = np.sort(df["date"].unique())
    total = len(unique_dates)
    min_train = int(total * min_train_frac)

    # Space origins evenly from min_train to (total - test_horizon)
    max_origin = total - test_horizon_days
    if max_origin <= min_train:
        # Not enough data for multiple origins; fall back to single split
        print(f"[WARNING] Only {total} dates, falling back to single origin")
        n_origins = 1
        origins = [min_train]
    else:
        origins = np.linspace(min_train, max_origin, n_origins, dtype=int)

    splits = []
    for origin in origins:
        train_dates = unique_dates[:origin]
        test_dates = unique_dates[origin:origin + test_horizon_days]
        # Use last 15% of training as validation
        val_split = int(len(train_dates) * 0.85)
        inner_train_dates = train_dates[:val_split]
        val_dates = train_dates[val_split:]

        train_df = df[df["date"].isin(train_dates)].reset_index(drop=True).copy()
        inner_train_df = df[df["date"].isin(inner_train_dates)].reset_index(drop=True).copy()
        val_df = df[df["date"].isin(val_dates)].reset_index(drop=True).copy()
        test_df = df[df["date"].isin(test_dates)].reset_index(drop=True).copy()

        if len(test_df) == 0:
            continue
        splits.append({
            "train_full": train_df,
            "train_inner": inner_train_df,
            "val": val_df,
            "test": test_df,
            "origin_date": str(unique_dates[origin])[:10],
        })
    return splits


# ============================================================================
# 5. BASELINES: Croston / SBA (walk-forward, leakage-free)
# ============================================================================

def _croston_walk_forward(series: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Walk-forward Croston: forecast[t] uses only series[0:t]."""
    n = len(series)
    forecast = np.zeros(n)
    q_est = None
    p_est = None
    t_last = -1

    for t in range(n):
        if q_est is None or p_est in (None, 0):
            forecast[t] = 0.0
        else:
            forecast[t] = q_est / p_est

        if series[t] > 0:
            interval = t - t_last if t_last >= 0 else t + 1
            t_last = t
            q_est = series[t] if q_est is None else alpha * series[t] + (1 - alpha) * q_est
            p_est = interval if p_est is None else alpha * interval + (1 - alpha) * p_est

    return forecast


def _sba_walk_forward(series: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    return (1.0 - alpha / 2.0) * _croston_walk_forward(series, alpha)


def _tsb_walk_forward(series: np.ndarray, alpha: float = 0.1,
                      beta: float = 0.1) -> np.ndarray:
    """Walk-forward TSB (Teunter-Syntetos-Babai, 2011).

    Extends Croston by tracking a separate obsolescence probability p_t that
    decays independently from the demand-size estimate q_t. This makes TSB
    more responsive than SBA when demand patterns are non-stationary.

    Forecast = p_t * q_t
    Update rules (after observing x_t):
        If x_t > 0: p_t = (1-beta)*p_t + beta*1; q_t = (1-alpha)*q_t + alpha*x_t
        If x_t = 0: p_t = (1-beta)*p_t;          q_t unchanged
    """
    n = len(series)
    forecast = np.zeros(n)
    # Initialise at first nonzero observation
    first_nz = next((i for i, v in enumerate(series) if v > 0), None)
    if first_nz is None:
        return forecast  # all zeros â€” return zeros

    q_est = series[first_nz]
    p_est = 1.0 / (first_nz + 1) if first_nz > 0 else 1.0  # initial demand probability

    for t in range(n):
        if t < first_nz:
            forecast[t] = 0.0
            continue
        forecast[t] = p_est * q_est
        if series[t] > 0:
            p_est = (1 - beta) * p_est + beta * 1.0
            q_est = (1 - alpha) * q_est + alpha * series[t]
        else:
            p_est = (1 - beta) * p_est
    return forecast


def tune_baseline_alpha(full_df, val_df, method="croston",
                         alpha_grid=(0.05, 0.1, 0.15, 0.2, 0.3),
                         beta_grid=(0.05, 0.1, 0.2)):
    """Grid-search smoothing parameter(s) on validation MASE."""
    y_val = val_df["sales"].values
    y_train_full = full_df["sales"].values
    best_score, best_params = np.inf, {"alpha": 0.1, "beta": 0.1}
    for a in alpha_grid:
        if method == "tsb":
            for b in beta_grid:
                full_df_sorted = full_df.sort_values(["series_id", "date"])
                preds = []
                for sid, g in full_df_sorted.groupby("series_id"):
                    preds.append(_tsb_walk_forward(g["sales"].values, alpha=a, beta=b))
                # align preds to val_df
                c_test, s_test, t_test = compute_baseline_predictions(full_df, val_df, alpha=a, beta=b)
                score = compute_mase(y_val, t_test, y_train_full)
                if score < best_score:
                    best_score, best_params = score, {"alpha": a, "beta": b}
        else:
            c, s, t = compute_baseline_predictions(full_df, val_df, alpha=a)
            pred = c if method == "croston" else s
            score = compute_mase(y_val, pred, y_train_full)
            if score < best_score:
                best_score, best_params = score, {"alpha": a, "beta": 0.1}
    return best_params

def compute_baseline_predictions(full_df: pd.DataFrame, test_df: pd.DataFrame,
                                  alpha: float = 0.1, beta: float = 0.1):
    """Walk-forward Croston / SBA / TSB predictions aligned to test_df rows."""
    full_df = full_df.sort_values(["series_id", "date"]).reset_index(drop=True)
    croston_map, sba_map, tsb_map = {}, {}, {}

    for sid, g in full_df.groupby("series_id"):
        vals = g["sales"].values
        c_fc = _croston_walk_forward(vals, alpha)
        s_fc = _sba_walk_forward(vals, alpha)
        t_fc = _tsb_walk_forward(vals, alpha, beta)
        for date, cf, sf, tf in zip(g["date"].values, c_fc, s_fc, t_fc):
            croston_map[(sid, date)] = cf
            sba_map[(sid, date)] = sf
            tsb_map[(sid, date)] = tf

    def _lookup(mapping):
        return test_df.apply(
            lambda r: mapping.get((r["series_id"], r["date"]), 0.0), axis=1
        ).values

    return _lookup(croston_map), _lookup(sba_map), _lookup(tsb_map)


def naive_probabilistic_baseline(full_df, val_df, test_df, alpha=0.1,
                                  quantiles=(0.10, 0.90)):
    """Croston point forecast +/- fixed empirical residual quantile."""
    croston_val, _, _ = compute_baseline_predictions(full_df, val_df, alpha=alpha)
    resid_val = val_df["sales"].values - croston_val
    q_lo, q_hi = np.quantile(resid_val, list(quantiles))

    croston_test, _, _ = compute_baseline_predictions(full_df, test_df, alpha=alpha)
    q10 = np.maximum(0, croston_test + q_lo)
    q50 = croston_test
    q90 = np.maximum(0, croston_test + q_hi)
    return q10, q50, q90



# ============================================================================
# 6. STAGE 1: Occurrence Classifier (LightGBM)
# ============================================================================

def train_occurrence_classifier(X_train, y_train, seed: int = 42) -> lgb.Booster:
    z_train = (y_train.values > 0).astype(int)
    dtrain = lgb.Dataset(X_train, label=z_train)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "seed": seed,
    }
    return lgb.train(params, dtrain, num_boost_round=300)


def predict_occurrence_prob(model: lgb.Booster, X) -> np.ndarray:
    return model.predict(X)


# ============================================================================
# 7. STAGE 2: BiGRU + Multi-Head Attention Quantile Net (WITH MEAN HEAD)
# ============================================================================

QUANTILES = [0.10, 0.50, 0.90]
WINDOW = 28

def get_seq_features(df: pd.DataFrame):
    base = ["sales"]
    optional = ["price", "promo", "day_sin", "day_cos", "day_of_week_sin", "day_of_week_cos", "month_of_year_sin", "month_of_year_cos"]
    return base + [c for c in optional if c in df.columns and df[c].nunique() > 1]

class SequenceDataset(Dataset):
    """Fixed-length lookback sequences. Target is log1p(sales[t]).
    Sequences end at t-1 (no leakage of the target into its input)."""

    def __init__(self, df, window: int = WINDOW, only_positive_target: bool = True):
        self.samples = []
        df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
        seq_features = get_seq_features(df)
        for sid, g in df.groupby("series_id"):
            g = g.reset_index(drop=True)
            vals = g[seq_features].values.astype(np.float32)
            sales = g["sales"].values.astype(np.float32)
            for t in range(window, len(g)):
                target = sales[t]
                if only_positive_target and target <= 0:
                    continue
                seq = vals[t - window:t, :]
                self.samples.append((seq, np.log1p(target)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return torch.from_numpy(seq), torch.tensor(target, dtype=torch.float32)


class TemporalAttentionQuantileNet(nn.Module):
    """BiGRU + multi-head self-attention with BOTH quantile heads AND
    a dedicated mean head for point-forecast optimality."""

    def __init__(self, n_features: int, hidden: int = 32,
                 n_heads: int = 4, n_quantiles: int = 3):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=2 * hidden, num_heads=n_heads,
                                           batch_first=True)
        self.norm = nn.LayerNorm(2 * hidden)
        # Quantile heads (P10, P50, P90 in log1p space)
        self.quantile_head = nn.Sequential(
            nn.Linear(2 * hidden, 32),
            nn.ReLU(),
            nn.Linear(32, n_quantiles),
        )
        # Mean head (conditional mean in log1p space) â€” NEW in v4
        self.mean_head = nn.Sequential(
            nn.Linear(2 * hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h, _ = self.gru(x)
        attn_out, _ = self.attn(h, h, h)
        h = self.norm(h + attn_out)
        pooled = h.mean(dim=1)
        quantiles = self.quantile_head(pooled)
        mean_est = self.mean_head(pooled).squeeze(-1)
        return quantiles, mean_est


def combined_loss(quantile_preds, mean_pred, target,
                  quantiles=QUANTILES, mean_weight: float = 0.5):
    """Combined pinball + MSE loss.
    mean_weight controls the trade-off: higher = better point forecast,
    lower = better calibrated intervals."""
    # Pinball loss for quantiles
    pinball_losses = []
    for i, q in enumerate(quantiles):
        err = target - quantile_preds[:, i]
        pinball_losses.append(torch.max((q - 1) * err, q * err))
    pinball = torch.stack(pinball_losses, dim=1).sum(dim=1).mean()

    # MSE loss for mean head
    mse = torch.nn.functional.mse_loss(mean_pred, target)

    return pinball + mean_weight * mse


def _run_epoch_v4(model, loader, opt, device, train: bool, mean_weight: float = 0.5):
    """Training/eval epoch with the combined loss."""
    model.train() if train else model.eval()
    total_loss = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for seq, target in loader:
            seq, target = seq.to(device), target.to(device)
            if train:
                opt.zero_grad()
            quantile_preds, mean_pred = model(seq)
            loss = combined_loss(quantile_preds, mean_pred, target,
                                 mean_weight=mean_weight)
            if train:
                loss.backward()
                opt.step()
            total_loss += loss.item() * seq.size(0)
            n += seq.size(0)
    return total_loss / max(n, 1)


def train_stage2(train_df, val_df=None, seed: int = 42, max_epochs: int = 60,
                  patience: int = 6, batch_size: int = 64, lr: float = 1e-3,
                  device=None, verbose: bool = True, mean_weight: float = 0.5, window: int = 28):
    """Train Stage 2 with early stopping on validation loss."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SequenceDataset(train_df, only_positive_target=True, window=window)
    if len(train_ds) == 0:
        raise ValueError("No positive-demand training sequences found.")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_df is not None:
        val_ds = SequenceDataset(val_df, only_positive_target=True, window=window)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    n_feats = len(get_seq_features(train_df))
    model = TemporalAttentionQuantileNet(n_features=n_feats).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = np.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        train_loss = _run_epoch_v4(model, train_loader, opt, device, train=True,
                                    mean_weight=mean_weight)

        if val_loader is not None:
            val_loss = _run_epoch_v4(model, val_loader, opt, device, train=False,
                                     mean_weight=mean_weight)
            improved = val_loss < best_val - 1e-4
            if improved:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epochs_no_improve == 0):
                print(f"  [Stage2] epoch {epoch+1}/{max_epochs}  train={train_loss:.4f}  "
                      f"val={val_loss:.4f}  best_val={best_val:.4f}  no_improve={epochs_no_improve}")

            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  [Stage2] early stopping at epoch {epoch+1}")
                break
        else:
            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                print(f"  [Stage2] epoch {epoch+1}/{max_epochs}  train={train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_stage2(model, device, df, window: int = WINDOW):
    """Predict quantiles AND mean for every row in df."""
    model.eval()
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    q_out = np.zeros((len(df), len(QUANTILES)), dtype=np.float32)
    m_out = np.zeros(len(df), dtype=np.float32)
    seq_features = get_seq_features(df)
    n_feats = len(seq_features)

    with torch.no_grad():
        for sid, g in df.groupby("series_id"):
            idx = g.index.values
            g = g.reset_index(drop=True)
            vals = g[seq_features].values.astype(np.float32)
            n = len(g)
            for local_t in range(n):
                start = local_t - window
                if start < 0:
                    seq = np.zeros((window, n_feats), dtype=np.float32)
                    avail = vals[0:local_t]
                    if len(avail) > 0:
                        seq[-len(avail):] = avail
                else:
                    seq = vals[start:local_t]
                seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)
                q_log, m_log = model(seq_t)
                q_pred = np.expm1(q_log.cpu().numpy()[0])
                q_pred = np.sort(q_pred)
                q_out[idx[local_t]] = np.maximum(0, q_pred)
                m_out[idx[local_t]] = max(0, np.expm1(m_log.cpu().numpy()[0]))

    return q_out, m_out


# ============================================================================
# 8. STAGE 3: Coupled Output + Conformal Calibration
# ============================================================================

def optimize_threshold(y_true_occ, p_hat, method: str = "youden"):
    """Optimize the occurrence gate threshold."""
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true_occ, p_hat)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])
    else:
        best_thresh, best_f1 = 0.5, -1
        for thresh in np.linspace(0.05, 0.95, 37):
            f1 = f1_score(y_true_occ, (p_hat >= thresh).astype(int))
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        return float(best_thresh)


def coupled_output_hard(p_hat, quantile_preds, mean_preds, theta: float):
    """HARD coupling: binary gate applied to both point forecast and intervals."""
    gate = (p_hat >= theta).astype(float)
    q10 = gate * quantile_preds[:, 0]
    q50 = gate * quantile_preds[:, 1]
    q90 = gate * quantile_preds[:, 2]
    point = gate * mean_preds
    return point, q10, q50, q90, gate


def coupled_output_soft(p_hat, quantile_preds, mean_preds, theta: float):
    """SOFT coupling: probability weighting applied to both point forecast and intervals."""
    gate = (p_hat >= theta).astype(float)
    q10 = p_hat * quantile_preds[:, 0]
    q50 = p_hat * quantile_preds[:, 1]
    q90 = p_hat * quantile_preds[:, 2]
    point = p_hat * mean_preds
    return point, q10, q50, q90, gate


def coupled_output_hybrid(p_hat, quantile_preds, mean_preds, theta: float):
    """HYBRID coupling: Soft point forecast + Hard intervals."""
    gate = (p_hat >= theta).astype(float)
    q10 = gate * quantile_preds[:, 0]
    q50 = gate * quantile_preds[:, 1]
    q90 = gate * quantile_preds[:, 2]
    point = p_hat * mean_preds
    return point, q10, q50, q90, gate


def coupled_output_mixture(p_hat, quantile_preds, mean_preds, tau=(0.10, 0.50, 0.90)):
    """Correct hurdle-mixture quantile: for target quantile level t,
    if t <= 1 - p_hat the mixture quantile is 0 (mass at zero dominates);
    otherwise it's the conditional-on-positive quantile at the
    rescaled level (t - (1 - p_hat)) / p_hat."""
    n = len(p_hat)
    out = np.zeros((n, len(tau)))
    for i in range(n):
        p = max(p_hat[i], 1e-6)
        for j, t in enumerate(tau):
            if t <= 1 - p:
                out[i, j] = 0.0
            else:
                rescaled = np.clip((t - (1 - p)) / p, 0.0, 1.0)
                out[i, j] = np.interp(rescaled, tau, quantile_preds[i])
    q10_mix, q50_mix, q90_mix = out[:, 0], out[:, 1], out[:, 2]
    point_mix = p_hat * mean_preds  # keep point forecast consistent with soft coupling
    gate = (p_hat > 0).astype(float)
    return point_mix, q10_mix, q50_mix, q90_mix, gate

# --- Conformal Calibration ---

def conformalize_quantiles_normalized(cal_q10, cal_q90, cal_y,
                                       target_coverage: float = 0.80,
                                       min_width: float = 1e-3):
    """Static split-conformal (normalized CQR)."""
    if len(cal_y) == 0:
        return 0.0
    alpha = 1.0 - target_coverage
    width = np.maximum(cal_q90 - cal_q10, min_width)
    scores = np.maximum((cal_q10 - cal_y) / width, (cal_y - cal_q90) / width)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    Qn = float(np.quantile(scores, level))
    return Qn


def apply_conformal_scale(q10, q90, gate, Qn: float, min_width: float = 1e-3):
    """Apply normalized conformal widening."""
    width = np.maximum(q90 - q10, min_width)
    q10_adj = np.where(gate > 0, np.maximum(0, q10 - Qn * width), q10)
    q90_adj = np.where(gate > 0, q90 + Qn * width, q90)
    return q10_adj, q90_adj


# --- Adaptive Conformal Inference ---

def adaptive_conformal_inference(q10_raw, q90_raw, y_true, gate,
                                  alpha: float = 0.20,
                                  gamma: float = 0.005,
                                  min_width: float = 1e-3):
    """Adaptive Conformal Inference (Gibbs & Candes, 2021)."""
    n = len(y_true)
    q10_adj = np.copy(q10_raw)
    q90_adj = np.copy(q90_raw)

    alpha_t = alpha
    Qn_t = 0.0

    scores_seen = []
    gated_indices = np.where(gate > 0)[0]

    for i, idx in enumerate(gated_indices):
        width = max(q90_raw[idx] - q10_raw[idx], min_width)
        q10_adj[idx] = max(0, q10_raw[idx] - Qn_t * width)
        q90_adj[idx] = q90_raw[idx] + Qn_t * width

        score = max((q10_raw[idx] - y_true[idx]) / width,
                    (y_true[idx] - q90_raw[idx]) / width)
        scores_seen.append(score)

        covered = (y_true[idx] >= q10_adj[idx]) and (y_true[idx] <= q90_adj[idx])
        err_t = 0.0 if covered else 1.0

        alpha_t = alpha_t + gamma * (alpha - err_t)
        alpha_t = np.clip(alpha_t, 0.01, 0.99)

        if len(scores_seen) >= 5:
            level = min(1.0, np.ceil((len(scores_seen) + 1) * (1 - alpha_t)) / len(scores_seen))
            Qn_t = max(0.0, float(np.quantile(scores_seen, level)))

    return q10_adj, q90_adj


def aggregated_aci(q10_raw, q90_raw, y_true, gate,
                   alpha: float = 0.20,
                   gammas: tuple = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
                   min_width: float = 1e-3):
    """AgACI â€” Aggregated Adaptive Conformal Inference."""
    K = len(gammas)
    alpha_ts = np.full(K, alpha, dtype=float)
    Qn_ts    = np.zeros(K, dtype=float)
    scores_seen = []
    weights  = np.ones(K, dtype=float)
    eta      = np.sqrt(8.0 * np.log(K))

    n = len(y_true)
    q10_adj = np.copy(q10_raw)
    q90_adj = np.copy(q90_raw)

    gated_indices = np.where(gate > 0)[0]

    for i, idx in enumerate(gated_indices):
        width = max(q90_raw[idx] - q10_raw[idx], min_width)
        w_sum = weights.sum()
        if w_sum == 0 or np.isnan(w_sum):
            weights = np.ones(K, dtype=float)
            w_sum = weights.sum()
        w_norm = weights / w_sum
        Qn_mix = float(np.dot(w_norm, Qn_ts))

        q10_adj[idx] = max(0, q10_raw[idx] - Qn_mix * width)
        q90_adj[idx] = q90_raw[idx] + Qn_mix * width

        score = max((q10_raw[idx] - y_true[idx]) / width,
                    (y_true[idx] - q90_raw[idx]) / width)
        scores_seen.append(score)

        for k, gamma_k in enumerate(gammas):
            cov_k = (y_true[idx] >= max(0, q10_raw[idx] - Qn_ts[k] * width)) and \
                    (y_true[idx] <= q90_raw[idx] + Qn_ts[k] * width)
            err_k = 0.0 if cov_k else 1.0

            alpha_ts[k] = np.clip(alpha_ts[k] + gamma_k * (alpha - err_k), 0.01, 0.99)

            if len(scores_seen) >= 5:
                level = min(1.0, np.ceil((len(scores_seen) + 1) *
                            (1 - alpha_ts[k])) / len(scores_seen))
                Qn_ts[k] = max(0.0, float(np.quantile(scores_seen, level)))

            weights[k] = np.clip(weights[k] * np.exp(-eta * err_k), 1e-15, 1e15)

    return q10_adj, q90_adj


# ============================================================================
# 9. PLAIN ENSEMBLE BASELINE & LGBM QUANTILE REGRESSOR
# ============================================================================

def train_lgbm_qr(X_train, y_train, seed: int = 42):
    """Train separate LightGBM models for 10th, 50th, and 90th percentiles."""
    models = {}
    for q in [0.10, 0.50, 0.90]:
        models[q] = lgb.train({
            "objective": "quantile",
            "alpha": q,
            "metric": "quantile",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbosity": -1,
            "seed": seed,
        }, lgb.Dataset(X_train, label=y_train), num_boost_round=300)
    return models

def predict_lgbm_qr(models, X_test):
    q10 = np.maximum(0, models[0.10].predict(X_test))
    q50 = np.maximum(0, models[0.50].predict(X_test))
    q90 = np.maximum(0, models[0.90].predict(X_test))
    return np.stack([q10, q50, q90], axis=1)

def train_plain_ensemble(X_train, y_train, seed: int = 42):
    lgb_model = lgb.train({
        "objective": "tweedie",
        "tweedie_variance_power": 1.22,
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "verbosity": -1,
        "seed": seed,
    }, lgb.Dataset(X_train, label=y_train), num_boost_round=450)

    xgb_model = xgb.XGBRegressor(
        n_estimators=350, learning_rate=0.02, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    return lgb_model, xgb_model


def predict_plain_ensemble(lgb_model, xgb_model, X_test, weight_lgb: float = 0.70):
    preds = weight_lgb * lgb_model.predict(X_test) + (1 - weight_lgb) * xgb_model.predict(X_test)
    return np.maximum(0, preds)


def tune_ensemble_weight(lgb_model, xgb_model, X_val, y_val):
    preds_lgb = lgb_model.predict(X_val)
    preds_xgb = xgb_model.predict(X_val)
    best_w, best_rmse = 0.70, np.inf
    for w in np.linspace(0.0, 1.0, 21):
        blend = np.maximum(0, w * preds_lgb + (1 - w) * preds_xgb)
        rmse = np.sqrt(np.mean((blend - y_val.values) ** 2))
        if rmse < best_rmse:
            best_rmse, best_w = rmse, w
    return float(best_w)


# ============================================================================
# 10. METRICS
# ============================================================================

def compute_mase(y_true, y_pred, y_train):
    scale = np.mean(np.abs(np.diff(y_train)))
    return np.mean(np.abs(y_true - y_pred)) / scale if scale != 0 else np.nan


def compute_rmsse(y_true, y_pred, y_train):
    scale = np.mean(np.diff(y_train) ** 2)
    return np.sqrt(np.mean((y_true - y_pred) ** 2) / scale) if scale != 0 else np.nan


def scaled_pinball_loss(y_true, q10, q50, q90, y_train):
    scale = np.mean(np.abs(np.diff(y_train)))
    if scale == 0:
        return np.nan
    losses = []
    for q, pred in zip(QUANTILES, [q10, q50, q90]):
        err = y_true - pred
        pl = np.mean(np.maximum(q * err, (q - 1) * err))
        losses.append(pl / scale)
    return float(np.mean(losses))


def compute_crps(y_true, q10, q50, q90):
    crps_vals = []
    for i in range(len(y_true)):
        y = y_true[i]
        for tau, q_val in zip(QUANTILES, [q10[i], q50[i], q90[i]]):
            indicator = 1.0 if y <= q_val else 0.0
            crps_vals.append(2 * abs((indicator - tau) * (q_val - y)))
    return float(np.mean(crps_vals))


def empirical_coverage(y_true, q10, q90):
    inside = (y_true >= q10) & (y_true <= q90)
    return float(np.mean(inside))


def stratified_coverage(y_true, q10, q90):
    occ_mask = y_true > 0
    overall = empirical_coverage(y_true, q10, q90)
    if occ_mask.sum() > 0:
        occurrence = empirical_coverage(y_true[occ_mask], q10[occ_mask], q90[occ_mask])
    else:
        occurrence = float("nan")
    return {"overall": overall, "occurrence": occurrence, "n_occurrence": int(occ_mask.sum())}

def decomposed_occurrence_coverage(y_true, q10, q90, gate):
    """Split occurrence-conditional coverage by whether the occurrence
    gate correctly fired on that day, to separate classifier recall
    from interval-calibration quality."""
    occ_mask = y_true > 0
    gated_hit  = occ_mask & (gate > 0)   # true sale, gate fired
    gated_miss = occ_mask & (gate == 0)  # true sale, gate missed (forced [0,0])
    out = {
        "classifier_recall_on_occ_days": float(gated_hit.sum() / max(1, occ_mask.sum())),
        "cov_occ_given_gate_fired": empirical_coverage(
            y_true[gated_hit], q10[gated_hit], q90[gated_hit]) if gated_hit.sum() > 0 else float("nan"),
        "n_gated_hit": int(gated_hit.sum()),
        "n_gated_miss": int(gated_miss.sum()),
    }
    return out

def diebold_mariano_test(errors_a, errors_b, power: int = 2):
    d = np.abs(errors_a) ** power - np.abs(errors_b) ** power
    n = len(d)
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    if d_var == 0:
        return 0.0, 1.0
    dm_stat = d_mean / np.sqrt(d_var / n)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return float(dm_stat), float(p_value)


def compute_spec(y_true, y_pred, holding_cost: float = 1.0,
                 backorder_cost: float = 9.0) -> float:
    under = backorder_cost * np.maximum(0.0, y_true - y_pred)
    over  = holding_cost  * np.maximum(0.0, y_pred - y_true)
    return float(np.mean(under + over))


def compute_mis(y_true, q10, q90, alpha: float = 0.10) -> float:
    width      = q90 - q10
    below_pen  = (2.0 / alpha) * np.maximum(0.0, q10 - y_true)
    above_pen  = (2.0 / alpha) * np.maximum(0.0, y_true - q90)
    return float(np.mean(width + below_pen + above_pen))


# ============================================================================
# 11. FULL PIPELINE: run_single_split
# ============================================================================

def run_single_split(train_inner, val_df, test_df, train_full, processed_df,
                     seed: int = 42,
                     test_horizon_days: int = 90,
                     seq_len: int = 28,
                     **kwargs):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  [Split] Inner train: {len(train_inner)} rows, Val: {len(val_df)}, Test: {len(test_df)}")

    # A2: Guarantee stable, identical sorting for combo_df and all test arrays
    test_df = test_df.sort_values(by=['series_id', 'date']).reset_index(drop=True)
    val_df = val_df.sort_values(by=['series_id', 'date']).reset_index(drop=True)

    feature_cols = get_feature_cols(processed_df)
    X_tr_inner, y_tr_inner = train_inner[feature_cols], train_inner["sales"]
    X_val, y_val = val_df[feature_cols], val_df["sales"]
    X_test, y_test = test_df[feature_cols], test_df["sales"]
    y_true = y_test.values
    y_train_full = train_full["sales"].values

    results = {}

    # --- Baselines ---
    print("Running baselines...")
    import time
    t0 = time.perf_counter()
    full_hist = processed_df[processed_df["date"] <= test_df["date"].max()]
    
    croston_alpha = tune_baseline_alpha(train_full, val_df, method="croston")["alpha"]
    sba_alpha     = tune_baseline_alpha(train_full, val_df, method="sba")["alpha"]
    tsb_params    = tune_baseline_alpha(train_full, val_df, method="tsb")
    
    # generate predictions separately
    croston_preds, _, _ = compute_baseline_predictions(full_hist, test_df, alpha=croston_alpha)
    _, sba_preds, _ = compute_baseline_predictions(full_hist, test_df, alpha=sba_alpha)
    _, _, tsb_preds = compute_baseline_predictions(full_hist, test_df, alpha=tsb_params["alpha"], beta=tsb_params["beta"])
    
    
    results["Croston"] = {
        "MASE":  compute_mase(y_true, croston_preds, y_train_full),
        "RMSSE": compute_rmsse(y_true, croston_preds, y_train_full),
        "SPEC":  compute_spec(y_true, croston_preds),
    }
    results["SBA"] = {
        "MASE":  compute_mase(y_true, sba_preds, y_train_full),
        "RMSSE": compute_rmsse(y_true, sba_preds, y_train_full),
        "SPEC":  compute_spec(y_true, sba_preds),
    }
    results["TSB"] = {
        "MASE":  compute_mase(y_true, tsb_preds, y_train_full),
        "RMSSE": compute_rmsse(y_true, tsb_preds, y_train_full),
        "SPEC":  compute_spec(y_true, tsb_preds),
    }

    # --- Naive probabilistic ---
    print("Running naive probabilistic...")
    naive_q10, naive_q50, naive_q90 = naive_probabilistic_baseline(full_hist, val_df, test_df)
    naive_cov = stratified_coverage(y_true, naive_q10, naive_q90)
    results["Naive Prob."] = {
        "MASE":  compute_mase(y_true, naive_q50, y_train_full),
        "RMSSE": compute_rmsse(y_true, naive_q50, y_train_full),
        "SPL":   scaled_pinball_loss(y_true, naive_q10, naive_q50, naive_q90, y_train_full),
        "CRPS":  compute_crps(y_true, naive_q10, naive_q50, naive_q90),
        "MIS":   compute_mis(y_true, naive_q10, naive_q90),
        "SPEC":  compute_spec(y_true, naive_q50),
        "Cov(all)": naive_cov["overall"],
        "Cov(occ)": naive_cov["occurrence"],
    }

    # --- Stage 1: Tree Ensembles ---
    print("Stage 1: Tree Ensembles...")
    y_tr_inner_occ = (y_tr_inner > 0).astype(int)
    occ_model = train_occurrence_classifier(X_tr_inner, y_tr_inner_occ, seed=seed)
    p_val = predict_occurrence_prob(occ_model, X_val)
    theta = optimize_threshold((y_val.values > 0).astype(int), p_val, method="youden")
    p_test = predict_occurrence_prob(occ_model, X_test)
    lgb_m, xgb_m = train_plain_ensemble(X_tr_inner, y_tr_inner, seed=seed)
    w = tune_ensemble_weight(lgb_m, xgb_m, X_val, y_val)
    plain_preds = predict_plain_ensemble(lgb_m, xgb_m, X_test, weight_lgb=w)

    # --- Hurdle-only ablation ---
    gate_test = (p_test >= theta).astype(float)
    hurdle_point = gate_test * plain_preds
    results["Hurdle-only"] = {
        "MASE":  compute_mase(y_true, hurdle_point, y_train_full),
        "RMSSE": compute_rmsse(y_true, hurdle_point, y_train_full),
        "SPEC":  compute_spec(y_true, hurdle_point),
    }
    results["Plain Ens."] = {
        "MASE":  compute_mase(y_true, plain_preds, y_train_full),
        "RMSSE": compute_rmsse(y_true, plain_preds, y_train_full),
        "SPEC":  compute_spec(y_true, plain_preds),
    }

    # --- Hurdle-LGBM-QR baseline ---
    print("Training Hurdle-LGBM Quantile Regressor Baseline...")
    lgbm_qr_models = train_lgbm_qr(X_tr_inner[y_tr_inner.values > 0], y_tr_inner[y_tr_inner.values > 0], seed=seed)
    q_preds_lgbm = predict_lgbm_qr(lgbm_qr_models, X_test)
    
    # Apply hard gate (hurdle) to LGBM QR
    q10_lgbm = gate_test * q_preds_lgbm[:, 0]
    q50_lgbm = gate_test * q_preds_lgbm[:, 1]
    q90_lgbm = gate_test * q_preds_lgbm[:, 2]
    
    cov_lgbm_qr = stratified_coverage(y_true, q10_lgbm, q90_lgbm)
    decomp_lgbm_qr = decomposed_occurrence_coverage(y_true, q10_lgbm, q90_lgbm, gate_test)
    results["Hurdle-LGBM-QR"] = {
        "MASE":  compute_mase(y_true, q50_lgbm, y_train_full),
        "RMSSE": compute_rmsse(y_true, q50_lgbm, y_train_full),
        "SPL":   scaled_pinball_loss(y_true, q10_lgbm, q50_lgbm, q90_lgbm, y_train_full),
        "CRPS":  compute_crps(y_true, q10_lgbm, q50_lgbm, q90_lgbm),
        "MIS":   compute_mis(y_true, q10_lgbm, q90_lgbm),
        "SPEC":  compute_spec(y_true, q50_lgbm),
        "Cov(all)": cov_lgbm_qr["overall"],
        "Cov(occ)": cov_lgbm_qr["occurrence"],
        "recall_on_occ": decomp_lgbm_qr["classifier_recall_on_occ_days"],
        "cov_occ_given_gate": decomp_lgbm_qr["cov_occ_given_gate_fired"],
    }

    # --- Stage 2: Deep Learning (TAN) ---
    print(f"  [Split] Training Stage 2 TAN (Hybrid)... SeqLen: {seq_len}")
    stage2_model = train_stage2(
        train_df=train_inner,
        val_df=val_df,
        seed=seed,
        max_epochs=CONFIG["stage2_max_epochs"],
        patience=CONFIG["stage2_patience"],
        window=seq_len
    )
    
    combo_df = pd.concat([train_full, test_df]).sort_values(by=['series_id', 'date']).reset_index(drop=True)
    q_preds, m_preds = predict_stage2(stage2_model, device, combo_df, window=seq_len)
    
    # Extract test rows (using vectorized exact key match, avoiding pandas index bugs)
    combo_df['_row_key'] = list(zip(combo_df['series_id'], combo_df['date']))
    test_key = set(zip(test_df['series_id'], test_df['date']))
    val_key  = set(zip(val_df['series_id'], val_df['date']))
    
    test_mask = combo_df['_row_key'].isin(test_key).values
    val_mask  = combo_df['_row_key'].isin(val_key).values
    combo_df = combo_df.drop(columns=['_row_key'])

    q_preds_test = q_preds[test_mask]
    m_preds_test = m_preds[test_mask]
    q_preds_val = q_preds[val_mask]
    
    # Enforce strict identical order for y_true based on combo_df
    y_true_rederived = combo_df.loc[test_mask, 'sales'].values
    assert np.allclose(y_true, y_true_rederived), "Test sorting mismatch!"

    # --- Stage 3: Hard coupling ---
    point_hard, q10_h, q50_h, q90_h, gate = coupled_output_hard(
        p_test, q_preds_test, plain_preds, theta
    )
    cov_hard = stratified_coverage(y_true, q10_h, q90_h)
    decomp_hard = decomposed_occurrence_coverage(y_true, q10_h, q90_h, gate)
    results["H-LGBM-TAN (hard)"] = {
        "MASE":  compute_mase(y_true, point_hard, y_train_full),
        "RMSSE": compute_rmsse(y_true, point_hard, y_train_full),
        "SPL":   scaled_pinball_loss(y_true, q10_h, q50_h, q90_h, y_train_full),
        "CRPS":  compute_crps(y_true, q10_h, q50_h, q90_h),
        "MIS":   compute_mis(y_true, q10_h, q90_h),
        "SPEC":  compute_spec(y_true, point_hard),
        "Cov(all)": cov_hard["overall"],
        "Cov(occ)": cov_hard["occurrence"],
        "recall_on_occ": decomp_hard["classifier_recall_on_occ_days"],
        "cov_occ_given_gate": decomp_hard["cov_occ_given_gate_fired"],
    }

    # --- Stage 3: Soft coupling ---
    point_soft, q10_s, q50_s, q90_s, gate = coupled_output_soft(
        p_test, q_preds_test, plain_preds, theta
    )
    cov_soft = stratified_coverage(y_true, q10_s, q90_s)
    decomp_soft = decomposed_occurrence_coverage(y_true, q10_s, q90_s, gate)
    results["H-LGBM-TAN (soft)"] = {
        "MASE":  compute_mase(y_true, point_soft, y_train_full),
        "RMSSE": compute_rmsse(y_true, point_soft, y_train_full),
        "SPL":   scaled_pinball_loss(y_true, q10_s, q50_s, q90_s, y_train_full),
        "CRPS":  compute_crps(y_true, q10_s, q50_s, q90_s),
        "MIS":   compute_mis(y_true, q10_s, q90_s),
        "SPEC":  compute_spec(y_true, point_soft),
        "Cov(all)": cov_soft["overall"],
        "Cov(occ)": cov_soft["occurrence"],
        "recall_on_occ": decomp_soft["classifier_recall_on_occ_days"],
        "cov_occ_given_gate": decomp_soft["cov_occ_given_gate_fired"],
    }

    # --- Stage 3: Hybrid coupling ---
    point_hybrid, q10_hyb, q50_hyb, q90_hyb, gate = coupled_output_hybrid(
        p_test, q_preds_test, plain_preds, theta
    )
    cov_hyb = stratified_coverage(y_true, q10_hyb, q90_hyb)
    decomp_hyb = decomposed_occurrence_coverage(y_true, q10_hyb, q90_hyb, gate)
    results["H-LGBM-TAN (hybrid)"] = {
        "MASE":  compute_mase(y_true, point_hybrid, y_train_full),
        "RMSSE": compute_rmsse(y_true, point_hybrid, y_train_full),
        "SPL":   scaled_pinball_loss(y_true, q10_hyb, q50_hyb, q90_hyb, y_train_full),
        "CRPS":  compute_crps(y_true, q10_hyb, q50_hyb, q90_hyb),
        "MIS":   compute_mis(y_true, q10_hyb, q90_hyb),
        "SPEC":  compute_spec(y_true, point_hybrid),
        "Cov(all)": cov_hyb["overall"],
        "Cov(occ)": cov_hyb["occurrence"],
        "recall_on_occ": decomp_hyb["classifier_recall_on_occ_days"],
        "cov_occ_given_gate": decomp_hyb["cov_occ_given_gate_fired"],
    }


    # --- Stage 3: Mixture coupling ---
    point_mixture, q10_mix, q50_mix, q90_mix, gate = coupled_output_mixture(
        p_test, q_preds_test, plain_preds, tau=(0.10, 0.50, 0.90)
    )
    cov_mix = stratified_coverage(y_true, q10_mix, q90_mix)
    decomp_mix = decomposed_occurrence_coverage(y_true, q10_mix, q90_mix, gate)
    results["H-LGBM-TAN (mixture)"] = {
        "MASE":  compute_mase(y_true, point_mixture, y_train_full),
        "RMSSE": compute_rmsse(y_true, point_mixture, y_train_full),
        "SPL":   scaled_pinball_loss(y_true, q10_mix, q50_mix, q90_mix, y_train_full),
        "CRPS":  compute_crps(y_true, q10_mix, q50_mix, q90_mix),
        "MIS":   compute_mis(y_true, q10_mix, q90_mix),
        "SPEC":  compute_spec(y_true, point_mixture),
        "Cov(all)": cov_mix["overall"],
        "Cov(occ)": cov_mix["occurrence"],
        "recall_on_occ": decomp_mix["classifier_recall_on_occ_days"],
        "cov_occ_given_gate": decomp_mix["cov_occ_given_gate_fired"],
    }

    # --- Static CQR calibration ---
    assert set(val_df['date']).isdisjoint(set(test_df['date'])), \
        "Calibration split overlaps test split - CQR would be invalid."

    cal_occ_mask = (val_df["sales"].values > 0)
    gate_val = (p_val >= theta).astype(float)
    if cal_occ_mask.sum() > 0:
        cal_occ_coverage_raw = empirical_coverage(
            val_df["sales"].values[cal_occ_mask],
            q_preds_val[cal_occ_mask, 0],
            q_preds_val[cal_occ_mask, 2],
        )
        Qn = conformalize_quantiles_normalized(
            q_preds_val[cal_occ_mask, 0], q_preds_val[cal_occ_mask, 2],
            val_df["sales"].values[cal_occ_mask],
            target_coverage=CONFIG["target_coverage"],
        )
    else:
        cal_occ_coverage_raw = float("nan")
        Qn = 0.0

    print(f"  [CQR] seed={seed}  Qn={Qn:+.6f}  cal_n={cal_occ_mask.sum()}  target={CONFIG['target_coverage']:.2f}")

    q10_cqr, q90_cqr = apply_conformal_scale(q10_s, q90_s, gate, Qn)
    cov_cqr = stratified_coverage(y_true, q10_cqr, q90_cqr)
    decomp_cqr = decomposed_occurrence_coverage(y_true, q10_cqr, q90_cqr, gate)
    results["H-LGBM-TAN + CQR"] = {
        "MASE":  compute_mase(y_true, point_soft, y_train_full),
        "RMSSE": compute_rmsse(y_true, point_soft, y_train_full),
        "SPL":   scaled_pinball_loss(y_true, q10_cqr, q50_s, q90_cqr, y_train_full),
        "CRPS":  compute_crps(y_true, q10_cqr, q50_s, q90_cqr),
        "MIS":   compute_mis(y_true, q10_cqr, q90_cqr),
        "SPEC":  compute_spec(y_true, point_soft),
        "Cov(all)": cov_cqr["overall"],
        "Cov(occ)": cov_cqr["occurrence"],
        "recall_on_occ": decomp_cqr["classifier_recall_on_occ_days"],
        "cov_occ_given_gate": decomp_cqr["cov_occ_given_gate_fired"],
    }

    # --- Adaptive Conformal Inference ---
    q10_aci, q90_aci = adaptive_conformal_inference(
        q10_cqr, q90_cqr, y_true, gate,
        alpha=1.0 - CONFIG["target_coverage"],
        gamma=CONFIG["aci_gamma"],
    )
    cov_aci = stratified_coverage(y_true, q10_aci, q90_aci)
    decomp_aci = decomposed_occurrence_coverage(y_true, q10_aci, q90_aci, gate)
    results["H-LGBM-TAN + ACI"] = {
        "MASE":  compute_mase(y_true, point_soft, y_train_full),
        "RMSSE": compute_rmsse(y_true, point_soft, y_train_full),
        "SPL":   scaled_pinball_loss(y_true, q10_aci, q50_s, q90_aci, y_train_full),
        "CRPS":  compute_crps(y_true, q10_aci, q50_s, q90_aci),
        "MIS":   compute_mis(y_true, q10_aci, q90_aci),
        "SPEC":  compute_spec(y_true, point_soft),
        "Cov(all)": cov_aci["overall"],
        "Cov(occ)": cov_aci["occurrence"],
        "recall_on_occ": decomp_aci["classifier_recall_on_occ_days"],
        "cov_occ_given_gate": decomp_aci["cov_occ_given_gate_fired"],
    }
    
    for g in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        q10_g, q90_g = adaptive_conformal_inference(
            q10_cqr, q90_cqr, y_true, gate,
            alpha=1.0 - CONFIG["target_coverage"],
            gamma=g,
        )
        cov_g = stratified_coverage(y_true, q10_g, q90_g)
        decomp_g = decomposed_occurrence_coverage(y_true, q10_g, q90_g, gate)
        results[f"H-LGBM-TAN + ACI (g={g})"] = {
            "Cov(occ)": cov_g["occurrence"],
            "recall_on_occ": decomp_g["classifier_recall_on_occ_days"],
            "cov_occ_given_gate": decomp_g["cov_occ_given_gate_fired"],
            "Cov(all)": cov_g["overall"],
        }

    # --- AgACI calibration ---
    q10_agaci, q90_agaci = aggregated_aci(
        q10_cqr, q90_cqr, y_true, gate,
        alpha=1.0 - CONFIG["target_coverage"],
    )
    cov_agaci = stratified_coverage(y_true, q10_agaci, q90_agaci)
    decomp_agaci = decomposed_occurrence_coverage(y_true, q10_agaci, q90_agaci, gate)
    results["H-LGBM-TAN + AgACI"] = {
        "MASE":  compute_mase(y_true, point_soft, y_train_full),
        "RMSSE": compute_rmsse(y_true, point_soft, y_train_full),
        "SPL":   scaled_pinball_loss(y_true, q10_agaci, q50_s, q90_agaci, y_train_full),
        "CRPS":  compute_crps(y_true, q10_agaci, q50_s, q90_agaci),
        "MIS":   compute_mis(y_true, q10_agaci, q90_agaci),
        "SPEC":  compute_spec(y_true, point_soft),
        "Cov(all)": cov_agaci["overall"],
        "Cov(occ)": cov_agaci["occurrence"],
        "recall_on_occ": decomp_agaci["classifier_recall_on_occ_days"],
        "cov_occ_given_gate": decomp_agaci["cov_occ_given_gate_fired"],
    }

    # --- Static CQR calibration on HYBRID backbone ---
    # Recompute Qn specifically for Hybrid quantiles
    if cal_occ_mask.sum() > 0:
        q10_hyb_val = gate_val * q_preds_val[:, 0]
        q90_hyb_val = gate_val * q_preds_val[:, 2]
        Qn_hyb = conformalize_quantiles_normalized(
            q10_hyb_val[cal_occ_mask], q90_hyb_val[cal_occ_mask],
            val_df["sales"].values[cal_occ_mask],
            target_coverage=CONFIG["target_coverage"],
        )
    else:
        Qn_hyb = 0.0

    q10_cqr_hyb, q90_cqr_hyb = apply_conformal_scale(q10_hyb, q90_hyb, gate, Qn_hyb)
    cov_cqr_hyb = stratified_coverage(y_true, q10_cqr_hyb, q90_cqr_hyb)
    decomp_cqr_hyb = decomposed_occurrence_coverage(y_true, q10_cqr_hyb, q90_cqr_hyb, gate)
    results["H-LGBM-TAN Hybrid + CQR"] = {
        "MASE":  compute_mase(y_true, point_hybrid, y_train_full),
        "RMSSE": compute_rmsse(y_true, point_hybrid, y_train_full),
        "SPL":   scaled_pinball_loss(y_true, q10_cqr_hyb, q50_hyb, q90_cqr_hyb, y_train_full),
        "CRPS":  compute_crps(y_true, q10_cqr_hyb, q50_hyb, q90_cqr_hyb),
        "MIS":   compute_mis(y_true, q10_cqr_hyb, q90_cqr_hyb),
        "SPEC":  compute_spec(y_true, point_hybrid),
        "Cov(all)": cov_cqr_hyb["overall"],
        "Cov(occ)": cov_cqr_hyb["occurrence"],
        "recall_on_occ": decomp_cqr_hyb["classifier_recall_on_occ_days"],
        "cov_occ_given_gate": decomp_cqr_hyb["cov_occ_given_gate_fired"],
    }

    # --- Adaptive Conformal Inference on HYBRID backbone ---
    q10_aci_hyb, q90_aci_hyb = adaptive_conformal_inference(
        q10_cqr_hyb, q90_cqr_hyb, y_true, gate,
        alpha=1.0 - CONFIG["target_coverage"],
        gamma=CONFIG["aci_gamma"],
    )
    cov_aci_hyb = stratified_coverage(y_true, q10_aci_hyb, q90_aci_hyb)
    decomp_aci_hyb = decomposed_occurrence_coverage(y_true, q10_aci_hyb, q90_aci_hyb, gate)
    results["H-LGBM-TAN Hybrid + ACI"] = {
        "MASE":  compute_mase(y_true, point_hybrid, y_train_full),
        "RMSSE": compute_rmsse(y_true, point_hybrid, y_train_full),
        "SPL":   scaled_pinball_loss(y_true, q10_aci_hyb, q50_hyb, q90_aci_hyb, y_train_full),
        "CRPS":  compute_crps(y_true, q10_aci_hyb, q50_hyb, q90_aci_hyb),
        "MIS":   compute_mis(y_true, q10_aci_hyb, q90_aci_hyb),
        "SPEC":  compute_spec(y_true, point_hybrid),
        "Cov(all)": cov_aci_hyb["overall"],
        "Cov(occ)": cov_aci_hyb["occurrence"],
        "recall_on_occ": decomp_aci_hyb["classifier_recall_on_occ_days"],
        "cov_occ_given_gate": decomp_aci_hyb["cov_occ_given_gate_fired"],
    }
    
    # --- Dump sample forecast for plot_ts.py ---
    target_series = "16014"
    if target_series in test_df["series_id"].astype(str).values:
        idx_mask = (test_df["series_id"].astype(str) == target_series).values
        sample_df = test_df.iloc[idx_mask].copy()
        sample_df["point_hybrid"] = point_hybrid[idx_mask]
        sample_df["q10_aci"] = q10_aci_hyb[idx_mask]
        sample_df["q90_aci"] = q90_aci_hyb[idx_mask]
        sample_df.to_csv("sample_forecast.csv", index=False)
        print("Saved sample_forecast.csv for item 16014!")


    # --- Diebold-Mariano tests ---
    err_soft   = y_true - point_soft
    err_hard   = y_true - point_hard
    err_hybrid = y_true - point_hybrid
    err_plain  = y_true - plain_preds

    dm_soft_vs_plain   = diebold_mariano_test(err_soft,   err_plain)
    dm_hard_vs_plain   = diebold_mariano_test(err_hard,   err_plain)
    dm_hybrid_vs_plain = diebold_mariano_test(err_hybrid, err_plain)
    dm_soft_vs_hybrid  = diebold_mariano_test(err_soft,   err_hybrid)
    dm_soft_vs_hard    = diebold_mariano_test(err_soft,   err_hard)

    results["_dm_tests"] = {
        "soft_vs_plain":   {"stat": dm_soft_vs_plain[0],   "p": dm_soft_vs_plain[1]},
        "hard_vs_plain":   {"stat": dm_hard_vs_plain[0],   "p": dm_hard_vs_plain[1]},
        "hybrid_vs_plain": {"stat": dm_hybrid_vs_plain[0], "p": dm_hybrid_vs_plain[1]},
        "soft_vs_hybrid":  {"stat": dm_soft_vs_hybrid[0],  "p": dm_soft_vs_hybrid[1]},
        "soft_vs_hard":    {"stat": dm_soft_vs_hard[0],    "p": dm_soft_vs_hard[1]},
    }
    
    return results


# 12. MAIN
# ============================================================================


from scipy.stats import bootstrap
def paired_bootstrap_diff(vals_a, vals_b, n_resamples=5000):
    """vals_a, vals_b: per-seed metric arrays for two models, same seeds/order."""
    import numpy as np
    diffs = np.array(vals_a) - np.array(vals_b)
    res = bootstrap((diffs,), np.mean, n_resamples=n_resamples, confidence_level=0.95)
    return {"mean_diff": float(np.mean(diffs)), "ci": (float(res.confidence_interval.low), float(res.confidence_interval.high))}

def run_full_experiment(dataset: str = "synthetic", **kwargs):
    import time
    t0_total = time.perf_counter()
    print(f"Loading dataset: {dataset}")
    raw_df = load_data(dataset=dataset, seed=CONFIG["seeds"][0], **kwargs)
    
    freq = "monthly" if dataset == "carparts" else "daily"
    print(f"Building features... (Frequency: {freq})")
    processed_df = build_features(raw_df, frequency=freq)
    
    processed_df.dropna(inplace=True)
    
    test_horizon = 6 if dataset == "carparts" else CONFIG["test_horizon_days"]
    seq_len = 12 if dataset == "carparts" else 28
    
    splits = rolling_origin_splits(
        processed_df,
        n_origins=CONFIG["n_origins"],
        test_horizon_days=test_horizon,
        min_train_frac=CONFIG["train_frac_initial"],
    )
    print(f"Rolling-origin CV: {len(splits)} origins")
    for i, s in enumerate(splits):
        print(f"  Origin {i+1}: train={len(s['train_full'])}, "
              f"val={len(s['val'])}, test={len(s['test'])}, "
              f"date={s['origin_date']}")
    print()

    # --- Run all combinations ---
    all_results = []
    seeds = CONFIG["seeds"]

    for origin_idx, split in enumerate(splits):
        for seed in seeds:
            print(f"--- Origin {origin_idx+1}/{len(splits)}, Seed {seed} ---")
            try:
                res = run_single_split(
                    train_inner=split["train_inner"],
                    val_df=split["val"],
                    test_df=split["test"],
                    train_full=split["train_full"],
                    processed_df=processed_df,
                    seed=seed,
                    test_horizon_days=test_horizon,
                    seq_len=seq_len,
                    verbose=False,
                )
                res["_origin"] = origin_idx
                res["_seed"] = seed
                all_results.append(res)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                continue

    total_time = time.perf_counter() - t0_total

    # --- Aggregate and report ---
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    model_names = ["Croston", "SBA", "TSB", "Naive Prob.", "Plain Ens.", "Hurdle-only",
                   "Hurdle-LGBM-QR", "H-LGBM-TAN (hard)", "H-LGBM-TAN (soft)", "H-LGBM-TAN (hybrid)",
                   "H-LGBM-TAN + CQR", "H-LGBM-TAN + ACI", "H-LGBM-TAN + AgACI",
                   "H-LGBM-TAN Hybrid + CQR", "H-LGBM-TAN Hybrid + ACI", "H-LGBM-TAN (mixture)"] + [f"H-LGBM-TAN + ACI (g={g})" for g in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]]
    metrics = ["MASE", "RMSSE", "SPL", "CRPS", "MIS", "SPEC", "Cov(all)", "Cov(occ)", "Qn",
               "val_occ_cov_raw"]

    summary_rows = []
    for model in model_names:
        row = {"Model": model}
        for metric in metrics:
            vals = [r[model][metric] for r in all_results
                    if model in r and metric in r[model]
                    and r[model][metric] is not None
                    and not np.isnan(r[model][metric])]
            if vals:
                row[f"{metric} (mean)"] = np.nanmean(vals)
                row[f"{metric} (std)"] = np.nanstd(vals)
                if metric == "MASE" and len(vals) > 1:
                    try:
                        from scipy.stats import bootstrap
                        ci = bootstrap((vals,), np.nanmean, n_resamples=2000, confidence_level=0.95)
                        row["MASE (ci_low)"] = ci.confidence_interval.low
                        row["MASE (ci_high)"] = ci.confidence_interval.high
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        pass
            else:
                row[f"{metric} (mean)"] = np.nan
                row[f"{metric} (std)"] = np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Print main table
    print("\n--- Point Forecast Metrics ---")
    point_cols = ["Model", "MASE (mean)", "MASE (std)", "MASE (ci_low)", "MASE (ci_high)", "RMSSE (mean)", "RMSSE (std)",
                  "SPEC (mean)", "SPEC (std)"]
    print(summary_df[[c for c in point_cols if c in summary_df.columns]].to_string(index=False))

    print("\n--- Probabilistic Forecast Metrics ---")
    prob_cols = ["Model", "SPL (mean)", "SPL (std)", "CRPS (mean)", "CRPS (std)",
                 "MIS (mean)", "MIS (std)", "Cov(all) (mean)", "Cov(all) (std)",
                 "Cov(occ) (mean)", "Cov(occ) (std)"]
    prob_df = summary_df[[c for c in prob_cols if c in summary_df.columns]]
    prob_df = prob_df.dropna(subset=[c for c in prob_df.columns if c != "Model"], how="all")
    print(prob_df.to_string(index=False))

    # --- CQR Qn diagnostic summary ---
    print("\n--- CQR Calibration Factor (Qn) Diagnostic ---")
    print("  Qn > 0  : intervals widened (under-covered val) â€” CQR active")
    print("  Qn = 0  : floor hit (over-covered val) â€” CQR is a no-op")
    print("  Qn < 0  : intervals narrowed (over-covered val) â€” CQR shrinks")
    qn_vals = [r["H-LGBM-TAN + CQR"].get("Qn", float("nan")) for r in all_results
               if "H-LGBM-TAN + CQR" in r]
    val_cov_vals = [r["H-LGBM-TAN + CQR"].get("val_occ_cov_raw", float("nan"))
                   for r in all_results if "H-LGBM-TAN + CQR" in r]
    if qn_vals:
        qn_arr = np.array(qn_vals)
        print(f"  Qn across {len(qn_arr)} runs: "
              f"mean={np.mean(qn_arr):+.6f}  "
              f"min={np.min(qn_arr):+.6f}  max={np.max(qn_arr):+.6f}  "
              f"n_zero={np.sum(qn_arr == 0.0)}  n_positive={np.sum(qn_arr > 0)}  "
              f"n_negative={np.sum(qn_arr < 0)}")
    if val_cov_vals:
        vc_arr = np.array([v for v in val_cov_vals if not np.isnan(v)])
        if len(vc_arr) > 0:
            print(f"  Val occ coverage (raw, before CQR): "
                  f"mean={np.mean(vc_arr):.4f}  target={CONFIG['target_coverage']:.2f}  "
                  f"({'over-covered' if np.mean(vc_arr) >= CONFIG['target_coverage'] else 'under-covered'} on val)")

    # --- Unconditional vs occurrence-conditional coverage warning ---
    soft_all = summary_df.loc[summary_df["Model"] == "H-LGBM-TAN (soft)",
                               "Cov(all) (mean)"].values
    hard_all = summary_df.loc[summary_df["Model"] == "H-LGBM-TAN (hard)",
                               "Cov(all) (mean)"].values
    if len(soft_all) > 0 and len(hard_all) > 0 and not np.isnan(soft_all[0]):
        print("\n--- Unconditional vs Occurrence-Conditional Coverage Trade-off ---")
        print(f"  Hard  Cov(all)={hard_all[0]:.4f}  "
              f"Cov(occ)={summary_df.loc[summary_df['Model']=='H-LGBM-TAN (hard)', 'Cov(occ) (mean)'].values[0]:.4f}")
        print(f"  Soft  Cov(all)={soft_all[0]:.4f}  "
              f"Cov(occ)={summary_df.loc[summary_df['Model']=='H-LGBM-TAN (soft)', 'Cov(occ) (mean)'].values[0]:.4f}")
        if soft_all[0] < 0.10:
            print(f"  *** WARNING: soft Cov(all) = {soft_all[0]:.4f} (< 10%). "
                  "Probability-weighted quantiles collapse coverage on zero-demand days. "
                  "This trade-off must be disclosed in the manuscript.")

    # --- DM tests ---
    def _dm_summary(key, label):
        vals = [r["_dm_tests"][key] for r in all_results
                if "_dm_tests" in r and key in r["_dm_tests"]]
        if not vals:
            return
        stats = np.array([d["stat"] for d in vals])
        ps    = np.array([d["p"]    for d in vals])
        # Stouffer's method to combine p-values across folds/seeds
        import scipy.stats as _st
        z_combined = np.sum(stats) / np.sqrt(len(stats))
        p_combined = 2 * (1 - _st.norm.cdf(abs(z_combined)))
        sig = "***" if p_combined < 0.001 else ("**" if p_combined < 0.01
              else ("*" if p_combined < 0.05 else "(ns)"))
        print(f"  {label:<30s}  avg_stat={np.mean(stats):+.3f}  "
              f"avg_p={np.mean(ps):.4f}  combined_p={p_combined:.4f} {sig}")
        print(f"    (negative stat = first model is BETTER; "
              f"p < 0.05 threshold for significance)")

    print("\n--- Diebold-Mariano Tests (squared errors) ---")
    print("  Negative DM stat means the first-named model has lower squared errors.")
    _dm_summary("soft_vs_plain",   "Soft   vs Plain")
    _dm_summary("hard_vs_plain",   "Hard   vs Plain")
    _dm_summary("hybrid_vs_plain", "Hybrid vs Plain")
    _dm_summary("soft_vs_hybrid",  "Soft   vs Hybrid  [coupling ablation]")
    _dm_summary("soft_vs_hard",    "Soft   vs Hard    [coupling ablation]")

    # --- Timing ---
    print("\n--- Timing (average per run) ---")
    timing_keys = ["baselines", "plain_ensemble", "stage1", "stage2_train", "stage2_infer"]
    for key in timing_keys:
        vals = [r["_timings"][key] for r in all_results if key in r.get("_timings", {})]
        if vals:
            print(f"  {key}: {np.mean(vals):.1f}s (std={np.std(vals):.1f}s)")
    print(f"  Total experiment: {total_time:.1f}s")

    # --- Key findings ---
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    soft_mase = summary_df.loc[summary_df["Model"] == "H-LGBM-TAN (soft)", "MASE (mean)"].values
    hybrid_mase = summary_df.loc[summary_df["Model"] == "H-LGBM-TAN (hybrid)", "MASE (mean)"].values
    hard_mase = summary_df.loc[summary_df["Model"] == "H-LGBM-TAN (hard)", "MASE (mean)"].values
    plain_mase = summary_df.loc[summary_df["Model"] == "Plain Ens.", "MASE (mean)"].values

    if len(soft_mase) > 0 and len(hard_mase) > 0 and len(hybrid_mase) > 0:
        print(f"  1. Hybrid coupling MASE: {hybrid_mase[0]:.4f} vs Soft: {soft_mase[0]:.4f} vs Hard: {hard_mase[0]:.4f}")
        if hybrid_mase[0] < hard_mase[0]:
            pct = (1 - hybrid_mase[0] / hard_mase[0]) * 100
            print(f"     -> Hybrid coupling improves MASE by {pct:.1f}% over Hard")

    cqr_cov = summary_df.loc[summary_df["Model"] == "H-LGBM-TAN + CQR", "Cov(occ) (mean)"].values
    aci_cov = summary_df.loc[summary_df["Model"] == "H-LGBM-TAN + ACI", "Cov(occ) (mean)"].values

    if len(cqr_cov) > 0 and len(aci_cov) > 0:
        print(f"  2. Occurrence coverage: CQR={cqr_cov[0]:.3f}, ACI={aci_cov[0]:.3f} (target={CONFIG['target_coverage']:.2f})")
        if aci_cov[0] > cqr_cov[0]:
            pp = (aci_cov[0] - cqr_cov[0]) * 100
            print(f"     -> ACI improves occurrence coverage by {pp:.1f}pp over static CQR")

    # Save results
    filename = "carparts_results.csv" if dataset == "carparts" else "results_summary.csv"
    summary_df.to_csv(filename, index=False)

    # --- B5: Paired Bootstrap CI ---
    print("\n--- B5: Paired Bootstrap CI on MASE ---")
    try:
        hurdle_vals = [r["Hurdle-only"]["MASE"] for r in all_results if "Hurdle-only" in r]
        hybrid_vals = [r["H-LGBM-TAN (hybrid)"]["MASE"] for r in all_results if "H-LGBM-TAN (hybrid)" in r]
        if hurdle_vals and hybrid_vals and len(hurdle_vals) == len(hybrid_vals):
            res1 = paired_bootstrap_diff(hurdle_vals, hybrid_vals)
            print(f"  Hurdle-only vs H-LGBM-TAN (hybrid): Mean Diff = {res1['mean_diff']:.4f}, 95% CI = [{res1['ci'][0]:.4f}, {res1['ci'][1]:.4f}]")
            
        lgbm_qr_vals = [r["Hurdle-LGBM-QR"]["MASE"] for r in all_results if "Hurdle-LGBM-QR" in r]
        mix_vals = [r["H-LGBM-TAN (mixture)"]["MASE"] for r in all_results if "H-LGBM-TAN (mixture)" in r]
        if lgbm_qr_vals and mix_vals and len(lgbm_qr_vals) == len(mix_vals):
            res2 = paired_bootstrap_diff(lgbm_qr_vals, mix_vals)
            print(f"  Hurdle-LGBM-QR vs H-LGBM-TAN (mixture): Mean Diff = {res2['mean_diff']:.4f}, 95% CI = [{res2['ci'][0]:.4f}, {res2['ci'][1]:.4f}]")
    except Exception as e:
        print("  Error computing B5 paired bootstrap:", e)

    print(f"\nResults saved to {filename}")

    return summary_df, all_results


# ============================================================================
# 13. ENTRY POINT (works in CLI, Jupyter, and Colab)
# ============================================================================

def _is_notebook():
    """Detect if running inside Jupyter/Colab."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell in ('ZMQInteractiveShell', 'Shell'):  # Jupyter / Colab
            return True
    except (ImportError, AttributeError):
        pass
    return False


if _is_notebook():
    # ---- NOTEBOOK / COLAB MODE ----
    # Edit these settings directly instead of using CLI arguments:
    NOTEBOOK_DATASET = "synthetic"   # "synthetic", "uci", "m5", or "csv"
    NOTEBOOK_CSV_PATH = None         # set if NOTEBOOK_DATASET == "csv"
    NOTEBOOK_M5_DIR = None           # set if NOTEBOOK_DATASET == "m5"

    # Optional overrides (set to None to use defaults from CONFIG):
    NOTEBOOK_SEEDS = None            # e.g. [42, 43, 44, 45, 46]
    NOTEBOOK_N_ORIGINS = None        # e.g. 3

    if NOTEBOOK_SEEDS is not None:
        CONFIG["seeds"] = NOTEBOOK_SEEDS
    if NOTEBOOK_N_ORIGINS is not None:
        CONFIG["n_origins"] = NOTEBOOK_N_ORIGINS

    kwargs = {}
    if NOTEBOOK_DATASET == "csv" and NOTEBOOK_CSV_PATH:
        kwargs["csv_path"] = NOTEBOOK_CSV_PATH
    elif NOTEBOOK_DATASET == "m5" and NOTEBOOK_M5_DIR:
        kwargs["m5_dir"] = NOTEBOOK_M5_DIR

    summary_df, all_results = run_full_experiment(
        dataset=NOTEBOOK_DATASET, **kwargs
    )

elif __name__ == "__main__":
    # ---- CLI MODE ----
    parser = argparse.ArgumentParser(description="H-LGBM-TAN v4 Experiment Runner")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "uci", "m5", "csv", "carparts"],
                        help="Dataset to use")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to CSV file (when --dataset csv)")
    parser.add_argument("--m5-dir", type=str, default=None,
                        help="Path to M5 data directory (when --dataset m5)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seed list (e.g. --seeds 42 43)")
    parser.add_argument("--n-origins", type=int, default=None,
                        help="Override number of rolling-origin folds")
    parser.add_argument("--n-series", type=int, default=None,
                        help="Override number of series (e.g., for UCI)")

    args = parser.parse_args()

    if args.seeds:
        CONFIG["seeds"] = args.seeds
    if args.n_origins:
        CONFIG["n_origins"] = args.n_origins
    if args.n_series:
        CONFIG["uci_n_series"] = args.n_series

    kwargs = {}
    if args.dataset == "csv":
        if not args.csv_path:
            parser.error("--csv-path required when --dataset csv")
        kwargs["csv_path"] = args.csv_path
    elif args.dataset == "m5":
        if not args.m5_dir:
            parser.error("--m5-dir required when --dataset m5")
        kwargs["m5_dir"] = args.m5_dir

    run_full_experiment(dataset=args.dataset, **kwargs)
