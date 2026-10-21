<div align="center">

# 🚚 Freight Rate Prediction
### Time-aware load rate forecasting with LightGBM

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-2E7D32?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8%2B-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)

**Author:** Umer Ahmed Baig Mughal <br>
**Programme:** MSc Robotics and Artificial Intelligence, ITMO University <br>
**GitHub:** [umerahmedbaig7](https://github.com/umerahmedbaig7)

</div>

Predict the posted rate (USD) of a freight load from its lane, distance, equipment type, weight and pickup date. The model is trained on Jan–Oct 2025 loads and predicts the 12,000 unlabeled loads from Nov–Dec 2025.

| | |
|---|---|
| **Model** | LightGBM on `log(rate / mile)`, averaged over 3 seeds |
| **Validation** | Rolling-origin (expanding-window) cross-validation: always train on the past and test on the next 2 months |
| **CV result** | MAE ≈ **$45**, MAPE ≈ **1.9%** on clean labels (baseline: MAE ≈ $82, MAPE ≈ 3.7%) |
| **Deliverables** | `validation_predictions.csv`, `scorer_results/candidate_december.png` |

---

## Repository structure

```
freight-rate-prediction/
├── data/
│   ├── train-test.csv                     # 48,000 labeled loads (Jan–Oct 2025)
│   ├── validation.csv                     # 12,000 loads to predict (Nov–Dec 2025)
│   ├── validation-predictions-template.csv
│   └── december-chart-inputs.csv          # 31 fixed-lane rows for the December chart
├── freight_rate_pipeline.py               # end-to-end pipeline (audit → CV → fit → predict)
├── score.py                               # provided scorer (validates files, draws chart)
├── requirements.txt
├── validation_predictions.csv             # load_id,predicted_rate
├── outputs/
│   ├── cv_results.csv                     # CV metrics averaged across folds
│   ├── cv_results_by_fold.csv             # CV metrics per fold
│   ├── data_quality_report.csv            # every data issue found + how it was handled
│   ├── december-chart-inputs.csv          # December file with predicted_rate filled in
│   ├── run_summary.json                   # parameters, features, metrics in one place
│   ├── fig_split_scheme.png               # train/validation split diagram
│   └── fig_cv_metrics.png                 # CV model comparison chart
└── scorer_results/
    └── candidate_december.png             # chart produced by score.py
```

> **File names:** the pipeline matches file names ignoring `-` vs `_`, so `train-test.csv` and `train_test.csv` are both found.

---

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/umerahmedbaig7/freight-rate-prediction.git
cd freight-rate-prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `matplotlib`, `lightgbm`.

## Run

```bash
python freight_rate_pipeline.py
```

This runs the whole workflow and writes all outputs listed above. Options:

| Flag | Effect |
|---|---|
| `--data-dir PATH` | Folder containing the CSVs (default `./data`) |
| `--out-dir PATH` | Folder for tables, figures and the summary JSON (default `./outputs`) |
| `--predictions-out PATH` | Where to write the final predictions (default `./validation_predictions.csv`) |
| `--fast` | Single seed inside CV only (final model still uses 3 seeds) |
| `--skip-cv` | Skip cross-validation and ablation entirely |
| `--skip-scorer` | Do not call `score.py` at the end |

To re-run the provided scorer on its own:

```bash
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions outputs/december-chart-inputs.csv
```

This checks that the submission has exactly 12,000 rows with columns `load_id,predicted_rate`, that the December file has 31 correct rows, and saves `scorer_results/candidate_december.png`.

---

## Approach

### 1. Validation strategy: time-based, not random

`train-test.csv` covers 1 Jan–31 Oct 2025 and `validation.csv` covers 1 Nov–31 Dec 2025, so the task is to forecast **future** months. A random split would put loads from the same days in both train and test and overstate accuracy.

I used three expanding-window folds, each predicting the following two months to mirror the Nov–Dec horizon:

| Fold | Train window | Test window | Train rows | Test rows |
|---|---|---|---|---|
| 1 | Jan–Jun | Jul–Aug | 28,806 | 9,671 |
| 2 | Jan–Jul | Aug–Sep | 33,718 | 9,429 |
| 3 | Jan–Aug | Sep–Oct | 38,477 | 9,523 |
| Final | Jan–Oct | Nov–Dec (no labels) | 48,000 | 12,000 |

**Leakage controls:** every statistic (weight median, per-bucket rate medians used to flag bad labels, baseline medians) is computed on the fold's training window only. Test rows are never removed; they are scored twice, on **all rows** and on **clean rows** (excluding corrupted labels).

### 2. Data-quality issues and handling

The full audit is in `outputs/data_quality_report.csv`.

| Issue | Count (train) | Handling |
|---|---|---|
| Corrupted target: rate spikes (×2–5) and drops (×0.2–0.5) | 677 rows (1.4%) | Detected as rate-per-mile deviating > 0.35 in log space from the equipment × distance-bucket median; **removed from training only** |
| Missing `weight` | 300 | Median imputation (training median) |
| Negative `weight` (sign errors) | 292 | `abs()` |
| `weight` censored at 47,500 | 1,204 | Added `weight_capped` flag |
| Missing `market_index` | 374 | Feature not used (see below) |
| Lat/lon not true city coordinates | all | Used as consistent per-city coordinates plus haversine distance, not literal geography |
| 8 cities in validation never seen in training (Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego); 736 unseen lanes | – | No city or lane ID features; geography is expressed through lat/lon, haversine and distance so new lanes still get sensible predictions |
| Duplicate `load_id` | 0 | None |

### 3. Features

`distance`, `log_dist`, `equip` (categorical), `weight`, `weight_capped`, `pickup_lat/lon`, `delivery_lat/lon`, `hav` (straight-line miles), `dist_ratio` (road ÷ straight-line circuity), `dow` (day of week).

**Deliberately excluded: `market_index` and `quote_signal`.** Both look useful in-sample, but their relationship with rate drifts month to month (for example, the monthly correlation of `quote_signal` with the rate residual flips between about +0.4 and −0.4, and the `market_index` monthly mean moves between about 0.89 and 1.30). The ablation below confirms they hurt out-of-time accuracy.

### 4. Model

LightGBM regressor predicting `log(rate / mile)`, converted back to dollars by `exp(pred) × distance`. Modelling rate per mile makes the target comparable across short and long hauls, and the log keeps errors proportional. Parameters are conservative (`num_leaves=15`, `min_child_samples=100`, `learning_rate=0.04`, 500 trees, row/column subsampling, L2 regularisation) and predictions are averaged over 3 seeds. The final model is refit on all 47,323 clean training rows.

### 5. Results (mean of 3 folds)

| Model | MAE $ (clean) | MAPE % (clean) | MAE $ (all) | MAPE % (all) |
|---|---|---|---|---|
| Baseline: equipment × distance-bucket median | 81.7 | 3.72 | 134.1 | 6.04 |
| LightGBM + `market_index` + `quote_signal` (rejected) | 78.9 | 3.60 | 131.3 | 5.84 |
| LightGBM, trained on uncleaned labels | 53.1 | 2.27 | 105.9 | 4.61 |
| **LightGBM final (stable features, cleaned labels)** | **44.8** | **1.89** | **97.7** | **4.23** |

Takeaways:
- Removing corrupted training labels cuts MAE by about $8 (53.1 → 44.8).
- Adding `market_index` / `quote_signal` makes the model about $34 worse out-of-time, so they are excluded.
- "All rows" error stays higher because the corrupted test labels are unpredictable by construction (RMSE on all rows is about $622–$631 for every model).

Figures: `outputs/fig_split_scheme.png`, `outputs/fig_cv_metrics.png`.

### 6. December chart

`scorer_results/candidate_december.png` is produced by the provided `score.py`. Inputs are fixed at Lexington → Fort Wayne, 360 miles, Dry Van, 32,000 lb; only the date changes. Predictions range from **$809 (Sunday)** to **$829 (Thursday)**.

---

## Author

<div align="center">

### Umer Ahmed Baig Mughal

🎓 **MSc Robotics and Artificial Intelligence**, ITMO University <br>
🔬 *Specialization: Machine Learning · Computer Vision · Human-Robot Interaction · Autonomous Systems · Robotic Motion Control*

[![GitHub](https://img.shields.io/badge/GitHub-umerahmedbaig7-181717?style=for-the-badge&logo=github)](https://github.com/umerahmedbaig7)

</div>
