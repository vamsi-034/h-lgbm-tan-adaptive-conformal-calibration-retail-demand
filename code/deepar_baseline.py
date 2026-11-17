import argparse
import json
import numpy as np
import pandas as pd
import time
import os

try:
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.torch.model.deepar import DeepAREstimator
    from gluonts.torch.distributions import NegativeBinomialOutput
    from gluonts.evaluation import Evaluator, make_evaluation_predictions
except ImportError:
    print("Error: gluonts not installed. Run 'pip install gluonts torch'")
    exit(1)

from H_LGBM_TAN_v4 import (
    load_data, classify_demand_syntetos_boylan, build_features,
    rolling_origin_splits, compute_mase, compute_rmsse,
    scaled_pinball_loss, compute_crps, empirical_coverage,
    stratified_coverage, compute_spec, compute_mis
)

def format_gluonts_dataset(df, target="sales", freq="D"):
    """Format DataFrame for GluonTS."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["series_id", "date"])
    
    real_cols = ["lag_7", "lag_14", "lag_28", 
                 "roll_mean_7", "roll_mean_28",
                 "day_of_week_sin", "day_of_week_cos"]
                 
    df[real_cols] = df.groupby("series_id")[real_cols].ffill().bfill().fillna(0)
    
    dataset = PandasDataset.from_long_dataframe(
        df, 
        target=target,
        item_id="series_id", 
        timestamp="date",
        freq=freq,
        feat_dynamic_real=real_cols
    )
    return dataset, real_cols


def run_deepar(dataset_name="synthetic", seed=42, max_epochs=50):
    print(f"Running DeepAR (NegBin) baseline on {dataset_name} data (seed {seed})...")
    
    df = load_data(dataset=dataset_name, seed=seed)
    cat_df = classify_demand_syntetos_boylan(df)
    processed_df = build_features(df)
    processed_df.dropna(subset=["lag_28"], inplace=True)
    
    test_horizon = 90
    splits = rolling_origin_splits(processed_df, n_origins=3, test_horizon_days=test_horizon)
    all_results = []
    
    for origin_idx, split_dict in enumerate(splits):
        train_df = split_dict["train_full"]
        val_df = split_dict["val"]
        test_df = split_dict["test"]
        print(f"\n--- Origin {origin_idx + 1}/3 ---")
        train_val_df = train_df.sort_values(["series_id", "date"])
        full_df = pd.concat([train_val_df, test_df]).sort_values(["series_id", "date"])
        
        train_ds, real_cols = format_gluonts_dataset(train_val_df)
        test_ds, _ = format_gluonts_dataset(full_df)
        
        estimator = DeepAREstimator(
            freq="D",
            prediction_length=test_horizon,
            context_length=test_horizon,
            distr_output=NegativeBinomialOutput(),
            num_layers=2,
            hidden_size=40,
            lr=3e-4,
            trainer_kwargs={"max_epochs": max_epochs, "accelerator": "cpu", "gradient_clip_val": 1.0}
        )
        
        t0 = time.time()
        predictor = estimator.train(train_ds, cache_data=True)
        train_time = time.time() - t0
        
        t0 = time.time()
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds,
            predictor=predictor,
            num_samples=100
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)
        infer_time = time.time() - t0
        
        q10_map, q50_map, q90_map = {}, {}, {}
        for forecast in forecasts:
            sid = forecast.item_id
            dates = pd.date_range(start=forecast.start_date.to_timestamp(), periods=test_horizon, freq="D")
            q10 = forecast.quantile(0.1)
            q50 = forecast.quantile(0.5)
            q90 = forecast.quantile(0.9)
            
            for d, q_lo, q_mid, q_hi in zip(dates, q10, q50, q90):
                d_str = d.strftime('%Y-%m-%d')
                q10_map[(sid, d_str)] = q_lo
                q50_map[(sid, d_str)] = q_mid
                q90_map[(sid, d_str)] = q_hi

        test_df_copy = test_df.copy()
        test_df_copy["date_str"] = test_df_copy["date"].astype(str)
        
        q10_pred = test_df_copy.apply(lambda r: q10_map.get((r["series_id"], r["date_str"]), 0.0), axis=1).values
        q50_pred = test_df_copy.apply(lambda r: q50_map.get((r["series_id"], r["date_str"]), 0.0), axis=1).values
        q90_pred = test_df_copy.apply(lambda r: q90_map.get((r["series_id"], r["date_str"]), 0.0), axis=1).values
        y_true = test_df["sales"].values
        y_train_full = train_val_df["sales"].values
        
        cov = stratified_coverage(y_true, q10_pred, q90_pred)
        
        origin_metrics = {
            "origin": origin_idx,
            "seed": seed,
            "MASE": compute_mase(y_true, q50_pred, y_train_full),
            "RMSSE": compute_rmsse(y_true, q50_pred, y_train_full),
            "SPL": scaled_pinball_loss(y_true, q10_pred, q50_pred, q90_pred, y_train_full),
            "CRPS": compute_crps(y_true, q10_pred, q50_pred, q90_pred),
            "MIS": compute_mis(y_true, q10_pred, q90_pred),
            "SPEC": compute_spec(y_true, q50_pred),
            "Cov(all)": cov["overall"],
            "Cov(occ)": cov["occurrence"],
            "train_time": train_time,
            "infer_time": infer_time
        }
        all_results.append(origin_metrics)
        print(f"Origin {origin_idx+1} Results:", origin_metrics)
        
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "uci"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    seeds = [42 + i for i in range(args.seeds)]
    all_runs = []
    
    for s in seeds:
        res = run_deepar(dataset_name=args.dataset, seed=s, max_epochs=args.epochs)
        all_runs.extend(res)
        
    df_res = pd.DataFrame(all_runs)
    df_res.to_csv(f"deepar_{args.dataset}_results.csv", index=False)
    
    print("\n--- DeepAR Final Results ---")
    summary = df_res.drop(columns=["origin", "seed"]).mean()
    print(summary)
