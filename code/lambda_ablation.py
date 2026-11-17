import argparse
import pandas as pd
import numpy as np
import torch
from H_LGBM_TAN_v4 import (
    load_data, build_features, rolling_origin_splits, train_stage2, predict_stage2,
    scaled_pinball_loss
)

def run_ablation(dataset="uci", seed=42):
    print(f"Running Lambda Ablation on {dataset} data (seed {seed})...")
    df = load_data(dataset=dataset, seed=seed)
    processed_df = build_features(df)
    processed_df.dropna(subset=["lag_28"], inplace=True)
    
    # Just use 1 origin to save time
    splits = rolling_origin_splits(processed_df, n_origins=1, test_horizon_days=90)
    split_dict = splits[0]
    
    train_inner = split_dict["train_inner"]
    val = split_dict["val"]
    test = split_dict["test"]
    
    lambdas = [0.1, 0.5, 0.9]
    results = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for lam in lambdas:
        print(f"\n--- Training Stage 2 with Lambda (mean_weight) = {lam} ---")
        model = train_stage2(train_inner, val_df=val, seed=seed, max_epochs=15, 
                             device=device, mean_weight=lam, verbose=False)
        
        q_preds, m_preds = predict_stage2(model, device, test)
        
        q10_pred = q_preds[:, 0]
        q50_pred = q_preds[:, 1]
        q90_pred = q_preds[:, 2]
        y_true = test["sales"].values
        y_train_full = pd.concat([train_inner, val])["sales"].values
        
        spl = scaled_pinball_loss(y_true, q10_pred, q50_pred, q90_pred, y_train_full)
        mse = np.mean((y_true - m_preds)**2)
        
        results.append({
            "Lambda": lam,
            "Scaled Pinball Loss": spl,
            "MSE": mse
        })
        
    res_df = pd.DataFrame(results)
    print("\n--- Ablation Results ---")
    print(res_df)
    res_df.to_csv("lambda_ablation_results.csv", index=False)

if __name__ == "__main__":
    run_ablation()
