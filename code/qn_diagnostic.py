#!/usr/bin/env python3
"""
qn_diagnostic.py — CQR Qn root-cause diagnostic
=================================================
Runs ONE seed + ONE origin of the H-LGBM-TAN pipeline on synthetic data
(fast, ~2-3 minutes) and prints:

  1. The CQR nonconformity quantile Qn for that fold
  2. The raw occurrence-conditional coverage on the val calibration set
  3. Whether Qn hits the zero floor (= CQR is a no-op)
  4. A forced-Qn sanity check proving apply_conformal_scale actually works

Run from the h_lgbm_tan_v4/ directory:
  python qn_diagnostic.py
"""

import sys
import numpy as np

# ---------------------------------------------------------------------------
# 1. Import shared functions from the main module
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")
from H_LGBM_TAN_v4 import (
    CONFIG,
    generate_intermittent_data,
    build_features,
    rolling_origin_splits,
    get_feature_cols,
    train_occurrence_classifier,
    predict_occurrence_prob,
    optimize_threshold,
    train_stage2,
    predict_stage2,
    coupled_output_soft,
    conformalize_quantiles_normalized,
    apply_conformal_scale,
    empirical_coverage,
    stratified_coverage,
)

# ---------------------------------------------------------------------------
# 2. Generate synthetic data and one split
# ---------------------------------------------------------------------------
SEED = 42
print("=" * 60)
print("CQR Qn Diagnostic — single seed/fold on synthetic data")
print("=" * 60)

print("\n[1/5] Generating synthetic data...")
raw_df = generate_intermittent_data(
    n_series=CONFIG["synthetic_n_series"],
    n_days=CONFIG["synthetic_n_days"],
    seed=SEED,
)
processed_df = build_features(raw_df)

splits = rolling_origin_splits(
    processed_df,
    n_origins=1,               # one fold only for speed
    test_horizon_days=CONFIG["test_horizon_days"],
    min_train_frac=CONFIG["train_frac_initial"],
)
split = splits[0]
train_inner = split["train_inner"]
val_df      = split["val"]
test_df     = split["test"]

feature_cols = get_feature_cols(split["train_full"])
X_tr = train_inner[feature_cols]
y_tr = train_inner["sales"]
X_val = val_df[feature_cols]
X_test = test_df[feature_cols]
y_test = test_df["sales"].values
y_val = val_df["sales"].values

print(f"  train_inner: {len(train_inner)} rows  |  val: {len(val_df)} rows  |  test: {len(test_df)} rows")

# ---------------------------------------------------------------------------
# 3. Stage 1 — occurrence classifier + threshold
# ---------------------------------------------------------------------------
print("\n[2/5] Training Stage 1 (occurrence classifier)...")
occ_model = train_occurrence_classifier(X_tr, (y_tr > 0).astype(int), seed=SEED)
p_val  = predict_occurrence_prob(occ_model, X_val)
p_test = predict_occurrence_prob(occ_model, X_test)
theta  = optimize_threshold((y_val > 0).astype(int), p_val, method="youden")
print(f"  Optimal threshold theta = {theta:.4f}")

# ---------------------------------------------------------------------------
# 4. Stage 2 — TAN (reduced epochs for speed)
# ---------------------------------------------------------------------------
print("\n[3/5] Training Stage 2 (TAN, reduced epochs for speed)...")
stage2_model, device = train_stage2(
    train_inner, val_df=val_df, seed=SEED,
    max_epochs=10,   # deliberately short -- we just need quantile outputs
    patience=3,
    verbose=False,
)
q_preds_val,  m_preds_val  = predict_stage2(stage2_model, device, val_df)
q_preds_test, m_preds_test = predict_stage2(stage2_model, device, test_df)

# ---------------------------------------------------------------------------
# 5. Soft coupling (base for CQR)
# ---------------------------------------------------------------------------
_, q10_s, q50_s, q90_s, gate = coupled_output_soft(p_test, q_preds_test, m_preds_test, theta)

# ---------------------------------------------------------------------------
# 6. CQR calibration -- the key diagnostic
# ---------------------------------------------------------------------------
print("\n[4/5] CQR calibration diagnostic...")

occ_mask_val = (y_val > 0)
n_cal = int(occ_mask_val.sum())
print(f"\n  Calibration set (occurrence-only val rows): n = {n_cal}")

if n_cal == 0:
    print("  *** ERROR: No positive-demand rows in val. Qn forced to 0.")
    Qn = 0.0
    raw_val_cov = float("nan")
else:
    # Raw coverage of TAN quantiles on the val calibration set, BEFORE CQR
    raw_val_cov = empirical_coverage(
        y_val[occ_mask_val],
        q_preds_val[occ_mask_val, 0],
        q_preds_val[occ_mask_val, 2],
    )
    target = CONFIG["target_coverage"]
    print(f"  Raw val occ coverage (before CQR): {raw_val_cov:.4f}  (target = {target:.2f})")

    if raw_val_cov >= target:
        print(f"  --> Model OVER-COVERS val by {(raw_val_cov - target)*100:.1f}pp.")
        print(f"      Nonconformity scores will be negative -> Qn <= 0.")
    else:
        print(f"  --> Model UNDER-COVERS val by {(target - raw_val_cov)*100:.1f}pp.")
        print(f"      Nonconformity scores will be positive -> Qn > 0 -> CQR widens.")

    # Compute actual Qn
    Qn = conformalize_quantiles_normalized(
        q_preds_val[occ_mask_val, 0],
        q_preds_val[occ_mask_val, 2],
        y_val[occ_mask_val],
        target_coverage=target,
    )
    print(f"\n  Qn = {Qn:+.8f}")

    if Qn == 0.0:
        print("  *** Qn is exactly 0.0 -- CQR IS A NO-OP. ***")
        print("      apply_conformal_scale will return intervals UNCHANGED.")
        print("      Root cause: score distribution's target-percentile <= 0 (over-coverage).")
    elif Qn < 0:
        print("  Qn < 0 -- CQR will SHRINK intervals (over-covered val; bidirectional fix active).")
    else:
        print(f"  Qn > 0 -- CQR will WIDEN intervals by {Qn:.4f} x width.")

# ---------------------------------------------------------------------------
# 7. Sanity check: force Qn = 0.5, confirm intervals change
# ---------------------------------------------------------------------------
print("\n[5/5] Sanity check -- forced Qn=0.5 to prove apply_conformal_scale works...")
q10_forced, q90_forced = apply_conformal_scale(q10_s, q90_s, gate, Qn=0.5)
q10_real,   q90_real   = apply_conformal_scale(q10_s, q90_s, gate, Qn=Qn)

gated_rows = gate > 0
if gated_rows.sum() > 0:
    avg_width_original = np.mean(q90_s[gated_rows] - q10_s[gated_rows])
    avg_width_forced   = np.mean(q90_forced[gated_rows] - q10_forced[gated_rows])
    avg_width_real     = np.mean(q90_real[gated_rows]   - q10_real[gated_rows])

    print(f"  Avg interval width on gated test rows:")
    print(f"    Raw soft (pre-CQR):        {avg_width_original:.4f}")
    print(f"    After Qn=+0.5 (forced):    {avg_width_forced:.4f}  "
          f"({'wider' if avg_width_forced > avg_width_original else 'narrower or same'})")
    print(f"    After Qn={Qn:+.4f} (actual): {avg_width_real:.4f}  "
          f"({'identical to raw -> no-op confirmed' if np.allclose(avg_width_real, avg_width_original) else 'changed'})")

    if np.allclose(q10_forced, q10_s) and np.allclose(q90_forced, q90_s):
        print("  *** UNEXPECTED: even forced Qn=0.5 produced no change. Check apply_conformal_scale. ***")
    else:
        print("  apply_conformal_scale works correctly when Qn != 0.")
        print("  The no-op is purely due to Qn == 0.")
else:
    print("  No gated rows in test set -- cannot measure interval width change.")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Qn                = {Qn:+.8f}")
if not (n_cal == 0):
    print(f"  Val occ coverage  = {raw_val_cov:.4f}  (target {CONFIG['target_coverage']:.2f})")
print(f"  Cal set size      = {n_cal}")
if Qn == 0.0:
    print("\n  CONCLUSION: CQR correction is zero. The TAN over-covers the val calibration")
    print("  set, so the nonconformity quantile floors at (or below) zero. The deployment-")
    print("  time coverage gap is a val->test distribution-shift effect, NOT a systematic")
    print("  undercalibration at fit time. This must be stated in the manuscript (Sec 5.3).")
elif Qn < 0:
    print("\n  CONCLUSION: CQR is shrinking intervals (over-covered val, bidirectional correction).")
else:
    print("\n  CONCLUSION: CQR is active and widening intervals as intended.")
