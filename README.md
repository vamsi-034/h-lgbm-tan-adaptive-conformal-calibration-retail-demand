# H-LGBM-TAN: Hybrid LightGBM–Temporal Attention Network for Intermittent Retail Demand

Implementation repository for the hybrid forecasting model combining two-stage hurdle LightGBM classification, BiGRU temporal attention network regression, and adaptive conformal prediction for zero-inflated, lumpy retail demand.

---

## Repository Structure

```
.
├── code/                      # Model architectures, training pipelines, and evaluation scripts
│   ├── H_LGBM_TAN_v4.py       # Main model pipeline (Hurdle LightGBM + BiGRU-TAN + Calibration)
│   ├── deepar_baseline.py     # DeepAR comparative baseline
│   ├── tft_baseline.py        # Temporal Fusion Transformer baseline
│   ├── m5_benchmark.py        # Multi-scale M5 Walmart benchmark runner
│   ├── lambda_ablation.py     # Coupling parameter sensitivity experiments
│   ├── qn_diagnostic.py       # Conformal quantile calibration diagnostics
│   ├── plot_ts.py             # 60-day illustrative forecast sequence generator
│   └── generate_figures.py    # Figure reproduction suite
├── data/                      # Processed datasets and sample forecast outputs
│   ├── uci_online_retail_processed.csv
│   ├── carparts.csv
│   └── sample_forecast.csv
├── results/                   # Tabulated experimental evaluations and diagnostic CSVs
│   ├── uci_results.csv
│   ├── carparts_results.csv
│   ├── m5_results.csv
│   ├── combined_results.csv
│   ├── results_summary.csv
│   └── stage1_diagnostics_metrics.csv
├── figures/                   # Publication-grade figures and diagnostic plots
│   ├── fig1_architecture.png
│   ├── fig2_workflow.png
│   ├── fig3_stage1_roc_pr.png
│   ├── fig4_occurrence_confusion_matrix.png
│   ├── fig5_data_sparsity.jpg
│   ├── fig6_m5_scalability.png
│   ├── fig7_timeseries_forecast.png
│   └── supplementary/
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Installation

```bash
git clone https://github.com/vamsi-034/h-lgbm-tan-adaptive-conformal-calibration-retail-demand.git
cd h-lgbm-tan-adaptive-conformal-calibration-retail-demand
pip install -r requirements.txt
```

---

## Execution Guide

### 1. Main Experiment (UCI Retail Benchmark)
```bash
python code/H_LGBM_TAN_v4.py
```

### 2. M5 Scalability Benchmark
```bash
python code/m5_benchmark.py
```

### 3. Generate Diagnostic and Manuscript Figures
```bash
python code/generate_figures.py
python code/plot_ts.py
```

---

## License
MIT License
