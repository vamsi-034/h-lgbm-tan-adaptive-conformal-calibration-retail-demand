# H-LGBM-TAN: Hybrid LightGBM–Temporal Attention Network with Adaptive Conformal Calibration for Intermittent Retail Demand Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Model: LightGBM](https://img.shields.io/badge/model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end framework combining **two-stage hurdle LightGBM classification**, **Bidirectional GRU Temporal Attention Network (BiGRU-TAN)** sequence regression, and **Adaptive Conformal Inference (ACI / CQR)** calibration for probabilistic forecasting of sparse, zero-inflated, and lumpy retail demand.

---

## 📖 Topic Overview: The Intermittent Retail Demand Problem

Retail supply chains frequently encounter products characterized by **intermittent and lumpy demand**—inventory items (e.g., replacement automotive parts, slow-moving electronics, industrial hardware, or wholesale retail items) where demand arrives sporadically and sales volume is zero on the majority of days.

```
Demand Volume
  ▲
  │                                   ▲ (4,000 unit spike)
  │                                   │
  │            ▲ (500 unit spike)     │
  │            │                      │
  │───0──0──0──┼──0──0──0──0──0──0──0──┼──0──0──0──0──► Time (Days)
     (70%+ Zero-Demand Periods)
```

According to the classic **Syntetos–Boylan classification grid**, demand is categorized by two fundamental properties:
1. **Average Demand Interval ($p$)**: The average number of periods between successive non-zero transactions ($p \ge 1.32$ denotes intermittent demand).
2. **Squared Coefficient of Variation ($CV^2$)**: The volatility of transaction quantities ($CV^2 \ge 0.49$ denotes lumpy demand).

When a time series exhibits both $p \ge 1.32$ and high $CV^2$ (in our retail benchmark, $CV^2 = 35.8$), forecasting becomes an extreme challenge due to two competing statistical characteristics:

* **Extreme Zero-Inflation (Sparsity)**: In our processed UCI retail benchmark, **71.75% of daily observations are exact zeros**. Standard single-stage regression models (linear models, plain neural networks, or standard gradient boosting) minimize continuous squared or absolute loss by collapsing predictions toward the dominant zero mass. This creates systemic under-prediction of genuine sales while generating false continuous micro-demand signals on zero days.
* **Heavy-Tailed Volatility (Lumpiness)**: When transactions occur, order sizes span orders of magnitude—from single-unit purchases to massive institutional bulk orders of 500 to 4,000+ units. Traditional parametric assumptions (Gaussian, Poisson) fail to accommodate these heavy tails.
* **Uncertainty Calibration Failure**: Reliable inventory management requires valid prediction intervals (e.g., nominal 80% coverage) to compute safety stocks. Standard quantile regression often under-covers during active demand surges or explodes under non-stationary shifts.

---

## 🚀 The H-LGBM-TAN Solution

H-LGBM-TAN decouples the forecasting problem into distinct occurrence, magnitude, and calibration stages:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: OCCURRENCE GATE & POINT ENSEMBLE (LightGBM)                            │
│  • Binary Classifier: Predicts P(Occurrence) on zero-inflated horizon            │
│  • Optimal Operating Threshold: Youden's J validation tuning (θ* ≈ 0.273)        │
│  • Point Ensemble: Fast gradient boosting baseline                               │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: TEMPORAL SEQUENCE MODELING & QUANTILE HEADS (BiGRU + Attention)         │
│  • Bidirectional GRU: Captures forward and backward temporal dependencies        │
│  • Temporal Attention Network (TAN): Multi-scale weighting across lag horizons    │
│  • Multi-Task Heads: Simultaneous conditional mean & pinball quantiles (q10, q90)│
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: COUPLING MECHANISMS & CONFORMAL CALIBRATION                             │
│  • Coupling: Hard, Soft, and Mixture gating: ŷ = P(Occ) · ŷ_active               │
│  • Conformalized Quantile Regression (CQR) & Adaptive Conformal Inference (ACI)  │
│  • Dynamically adjusts interval bounds to guarantee finite-sample (1-α) coverage │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Key Methodological Contributions

1. **Youden's $J$ Threshold Optimization**: Rather than applying a naive 0.5 decision cutoff that destroys sensitivity under 70%+ zeros, Stage 1 tunes $\theta^*$ via validation Youden's $J$ ($J = \text{TPR} - \text{FPR}$), yielding an optimal operating threshold ($\theta^* = 0.273 \pm 0.030$, $\text{AUC} = 0.808$).
2. **Multi-Head BiGRU-TAN**: Uses sequence history (rolling averages, cyclical calendar encodings, historical demand lags) to predict active demand magnitudes without contamination from zero-padded days.
3. **Conformal Coverage Guarantees**: Integrates online non-conformity updating (ACI) to adapt prediction intervals to distribution drift, ensuring rigorous $(1-\alpha)$ coverage across evaluation folds.
4. **Occurrence-Conditional Coverage Insights**: Reveals that while conformal calibration achieves nominal 80% unconditional coverage across the full time horizon, occurrence-conditional coverage ($Cov(occ)$) reflects the physical upper boundary of extreme discrete order spikes (e.g., 4,000 units).

---

## 📊 Benchmark Datasets & Results Summary

The architecture was evaluated across three distinct intermittent benchmarks:

1. **UCI Online Retail II**: 500 intermittent series evaluated across 15 rolling-origin folds (3 origins × 5 seeds; $N = 270,000$ test evaluations). 71.75% zero share, $CV^2 = 35.8$.
2. **Automotive Car Parts**: Industrial spare-parts inventory dataset exhibiting erratic transaction intervals.
3. **M5 Walmart Competition**: Scalability benchmark evaluating 500 intermittent retail series.

### Representative Performance Comparison (UCI Retail)

| Model Family | Model Variant | MASE (Point) ↓ | SPL (80%) ↓ | Cov(all) | Cov(occ) | MIS ↓ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Statistical Baselines** | Croston's Method | 1.042 | — | — | — | — |
| | Syntetos-Boylan (SBA) | 1.018 | — | — | — | — |
| | Teunter-Syntetos-Babai (TSB)| 0.994 | — | — | — | — |
| **Deep Learning Baselines** | DeepAR | 0.982 | 0.412 | 0.741 | 0.512 | 84.2 |
| | Temporal Fusion Transformer| 0.965 | 0.389 | 0.762 | 0.534 | 78.6 |
| **Proposed Framework** | H-LGBM-TAN (Hard Coupling) | 0.941 | 0.364 | 0.798 | 0.589 | 64.2 |
| | H-LGBM-TAN (Mixture Coupling)| 0.936 | 0.358 | 0.801 | 0.598 | 61.8 |
| | **H-LGBM-TAN (Hybrid + ACI)** | **0.932** | **0.349** | **0.802** | **0.604** | **59.4** |

* **MASE**: Mean Absolute Scaled Error (lower is better; $< 1.0$ outperforms naive in-sample persistence).
* **SPL**: Scaled Pinball Loss (probabilistic sharpness).
* **Cov(all)**: Unconditional 80% prediction interval coverage (target = 0.80).
* **Cov(occ)**: Occurrence-conditional coverage on non-zero demand days.
* **MIS**: Mean Interval Score (evaluates both coverage accuracy and interval tightness).

---

## 📁 Repository Structure

```
.
├── code/                      # Python implementations, benchmarks, and figure generators
│   ├── H_LGBM_TAN_v4.py       # Core H-LGBM-TAN implementation (Stage 1 + Stage 2 + Stage 3)
│   ├── deepar_baseline.py     # DeepAR comparative baseline
│   ├── tft_baseline.py        # Temporal Fusion Transformer comparative baseline
│   ├── m5_benchmark.py        # M5 Walmart benchmark suite
│   ├── m5_run_500.py          # M5 500-series scalability runner
│   ├── m5_run_1000.py         # M5 1000-series scalability runner
│   ├── m5_run_2000.py         # M5 2000-series scalability runner
│   ├── lambda_ablation.py     # Coupling parameter lambda ablation experiments
│   ├── qn_diagnostic.py       # Conformal quantile calibration diagnostics
│   ├── plot_ts.py             # Publication 60-day illustrative forecast sequence generator
│   ├── generate_figures.py    # Figure generation pipeline
│   └── download_carparts.py   # Dataset helper for Car Parts benchmark
├── data/                      # Clean processed datasets for rapid out-of-the-box execution
│   ├── uci_online_retail_processed.csv   # Processed UCI retail dataset
│   ├── carparts.csv                      # Automotive spare parts dataset
│   └── sample_forecast.csv               # 60-day rolling test sequence (Stock Code 16014)
├── results/                   # Tabulated experimental metric files (CSV format)
│   ├── uci_results.csv                   # UCI fold-by-fold results across baselines and variants
│   ├── carparts_results.csv              # Car Parts benchmark evaluation metrics
│   ├── m5_results.csv                    # Walmart M5 benchmark metrics
│   ├── combined_results.csv              # Unified multi-dataset comparison
│   ├── results_summary.csv               # Aggregated metrics matching manuscript tables
│   ├── stage1_diagnostics_metrics.csv    # 15-fold Stage 1 classification diagnostics
│   ├── deepar_uci_results.csv            # DeepAR evaluation outputs
│   ├── tft_uci_results.csv               # TFT evaluation outputs
│   └── lambda_ablation_results.csv       # Coupling sensitivity results
├── figures/                   # Publication-ready diagnostic and architectural figures (300 DPI)
│   ├── fig1_architecture.png             # Figure 1: 3-Stage H-LGBM-TAN Architecture
│   ├── fig2_workflow.png                 # Figure 2: Experimental Methodology Pipeline
│   ├── fig3_stage1_roc_pr.png            # Figure 3: Stage 1 ROC and PR Diagnostic Curves
│   ├── fig4_occurrence_confusion_matrix.png # Figure 4: Pooled 270,000-sample Confusion Matrix
│   ├── fig5_data_sparsity.jpg            # Figure 5: Sparsity & Heavy-Tailed Volatility Distribution
│   ├── fig6_m5_scalability.png           # Figure 6: M5 Multi-Scale Benchmark & Log-SPL Stability
│   ├── fig7_timeseries_forecast.png      # Figure 7: 60-Day Illustrative Forecast Sequence
│   ├── fig7_timeseries_forecast.pdf      # Figure 7 (Vector PDF)
│   ├── architecture_diagram.html         # Interactive visual architecture diagram
│   └── supplementary/                    # Auxiliary ablation and metric bar charts
├── requirements.txt           # Python dependency specifications
└── README.md                  # Project documentation
```

---

## 🛠️ Quickstart & Execution

### 1. Installation

```bash
git clone https://github.com/vamsi-034/h-lgbm-tan-adaptive-conformal-calibration-retail-demand.git
cd h-lgbm-tan-adaptive-conformal-calibration-retail-demand
pip install -r requirements.txt
```

### 2. Run the Main H-LGBM-TAN Pipeline
Executes rolling-origin cross-validation on the UCI Retail dataset, evaluates coupling mechanisms, and computes conformal prediction intervals:

```bash
python code/H_LGBM_TAN_v4.py
```

### 3. Run Benchmark Baselines
```bash
python code/deepar_baseline.py
python code/tft_baseline.py
python code/m5_benchmark.py
```

### 4. Reproduce Figures
Generate all publication-grade figures:

```bash
# Generate architecture and diagnostic plots
python code/generate_figures.py

# Generate unclipped 60-day illustrative forecast plot (Figure 7)
python code/plot_ts.py
```

---

## 📜 Citation & License

This project is licensed under the [MIT License](LICENSE).
