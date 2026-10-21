#!/usr/bin/env python3
"""
Freight Rate Prediction Challenge - end-to-end pipeline
=======================================================

    python freight_rate_pipeline.py

Outputs
-------
  validation_predictions.csv                 -> submit (exactly: load_id,predicted_rate)
  outputs/december-chart-inputs.csv          -> December file with predicted_rate filled in
  scorer_results/candidate_december.png      -> produced by the provided score.py (goes in report)
  outputs/cv_results.csv                     -> temporal cross-validation table (for report)
  outputs/data_quality_report.csv            -> data-quality findings (for report + Loom)
  outputs/run_summary.json                   -> all key numbers in one place
  outputs/fig_split_scheme.png               -> time-based split figure
  outputs/fig_cv_metrics.png                 -> cross-validation metrics (unless --skip-cv)

Design summary:
  * Split      : Train/test data = Jan-Oct 2025, validation = Nov-Dec 2025,
                 so development uses rolling-origin folds that always predict the FUTURE.
  * Target     : log(rate / distance), converted back to dollars.
  * Cleaning   : abs() on negative weights, cap flag, median imputation, and training-time
                 removal of corrupted labels (rate spikes / drops).
  * Features   : distance, equipment, weight, geography (lat/lon, haversine, circuity),
                 day-of-week.
  * Excluded   : market_index level and quote_signal - both look predictive in-sample but their
                 relationship with rate is NOT stable over time; they hurt out-of-time accuracy
                 (shown in the CV ablation).
  * Model      : LightGBM (regularised), 3-seed average.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    raise SystemExit("lightgbm is required:  python -m pip install -r requirements.txt")

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
EXPECTED_ROWS = 12_000
WEIGHT_CAP = 47_500.0                       # weight is censored at this value in the data
LABEL_DEV_THRESHOLD = 0.35                  # |log-rate-per-mile deviation| above this => corrupted label
DIST_BINS = [0, 150, 300, 500, 800, 1200, 1800, 2500, np.inf]
EQUIP_MAP = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
SEEDS = (0, 1, 2)

LGB_PARAMS = dict(
    objective="regression",       # on log(rate per mile)
    learning_rate=0.04,
    n_estimators=500,
    num_leaves=15,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
    n_jobs=-1,
)

# Rolling-origin (expanding window) folds: always train on the past, predict the next 2 months.
CV_FOLDS = [
    ("2025-01-01", "2025-06-30", "2025-07-01", "2025-08-31"),
    ("2025-01-01", "2025-07-31", "2025-08-01", "2025-09-30"),
    ("2025-01-01", "2025-08-31", "2025-09-01", "2025-10-31"),
]

STABLE_FEATURES = [
    "distance", "log_dist", "equip", "weight", "weight_capped",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "hav", "dist_ratio", "dow",
]


# --------------------------------------------------------------------------------------
# File discovery / loading
# --------------------------------------------------------------------------------------
def _norm(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def find_file(target: str, search_dirs: list[Path]) -> Path:
    """Find e.g. 'train_test.csv' even if it is named 'train-test.csv' or sits next to the script."""
    for folder in search_dirs:
        if folder.is_dir():
            for path in folder.iterdir():
                if path.is_file() and _norm(path.name) == _norm(target):
                    return path
    raise SystemExit(
        f"Could not find '{target}'. Looked in: {', '.join(str(d) for d in search_dirs)}. "
        "Use --data-dir to point to the folder containing the CSV files."
    )


def load_inputs(data_dir: Path):
    here = Path(__file__).resolve().parent
    dirs = [data_dir, Path.cwd(), Path.cwd() / "data", here, here / "data"]
    paths = {
        "train": find_file("train_test.csv", dirs),
        "val": find_file("validation.csv", dirs),
        "template": find_file("validation_predictions_template.csv", dirs),
        "dec": find_file("december_chart_inputs.csv", dirs),
    }
    train = pd.read_csv(paths["train"], parse_dates=["date"])
    val = pd.read_csv(paths["val"], parse_dates=["date"])
    template = pd.read_csv(paths["template"])
    dec = pd.read_csv(paths["dec"])
    return train, val, template, dec, paths


# --------------------------------------------------------------------------------------
# Cleaning + features
# --------------------------------------------------------------------------------------
def haversine_miles(df: pd.DataFrame) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(df[c].astype(float)) for c in
                              ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def build_features(df: pd.DataFrame, weight_median: float) -> pd.DataFrame:
    """Stable, time-robust features only (no market_index / quote_signal)."""
    f = pd.DataFrame(index=df.index)
    f["distance"] = df["distance"].astype(float)
    f["log_dist"] = np.log(f["distance"])
    f["equip"] = df["equipment"].map(EQUIP_MAP).astype(int)
    w = df["weight"].abs()                                  # negative weights are sign errors
    f["weight"] = w.fillna(weight_median)                   # median imputation (train median)
    f["weight_capped"] = (w >= WEIGHT_CAP).astype(int)      # censored at 47,500
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        f[c] = df[c].astype(float)
    f["hav"] = haversine_miles(df)
    f["dist_ratio"] = f["distance"] / f["hav"].replace(0, np.nan)   # road circuity
    f["dow"] = df["date"].dt.dayofweek
    return f[STABLE_FEATURES]


def label_deviation(df: pd.DataFrame) -> pd.Series:
    """log(rate/mile) minus the median for the same equipment x distance bucket."""
    lrpm = np.log(df["posted_rate"] / df["distance"])
    groups = [df["equipment"], pd.cut(df["distance"], DIST_BINS)]
    return lrpm - lrpm.groupby(groups, observed=True).transform("median")


def corrupted_label_mask(df: pd.DataFrame) -> pd.Series:
    return label_deviation(df).abs() > LABEL_DEV_THRESHOLD


# --------------------------------------------------------------------------------------
# Data-quality audit
# --------------------------------------------------------------------------------------
def audit_data(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(issue, train_val, val_val, action):
        rows.append({"issue": issue, "train_test": train_val, "validation": val_val, "how_handled": action})

    add("Rows", len(train), len(val), "-")
    add("Date range", f"{train.date.min().date()} to {train.date.max().date()}",
        f"{val.date.min().date()} to {val.date.max().date()}",
        "Validation is strictly AFTER training -> time-based split + rolling-origin CV")
    add("Missing weight", int(train.weight.isna().sum()), int(val.weight.isna().sum()),
        "Median imputation (train median)")
    add("Missing market_index", int(train.market_index.isna().sum()), int(val.market_index.isna().sum()),
        "Not used as a model feature (unstable over time)")
    add("Negative weight (sign error)", int((train.weight < 0).sum()), int((val.weight < 0).sum()),
        "abs(); magnitudes match the normal range")
    add("Weight censored at 47,500", int((train.weight.abs() >= WEIGHT_CAP).sum()),
        int((val.weight.abs() >= WEIGHT_CAP).sum()), "Added weight_capped flag")
    mask = corrupted_label_mask(train)
    dev = label_deviation(train)
    add("Corrupted target (rate spike x2-5)", int((dev > LABEL_DEV_THRESHOLD).sum()), "n/a (no labels)",
        "Dropped from TRAINING only; metrics reported on all and on clean rows")
    add("Corrupted target (rate drop x0.2-0.5)", int((dev < -LABEL_DEV_THRESHOLD).sum()), "n/a (no labels)",
        "Dropped from TRAINING only")
    add("Corrupted-label share of train", f"{mask.mean():.2%}", "n/a", "-")
    h_tr, h_va = haversine_miles(train), haversine_miles(val)
    add("Road/straight-line distance ratio > 1.5 (short hauls)", int((train.distance / h_tr > 1.5).sum()),
        int((val.distance / h_va > 1.5).sum()), "Kept; circuity ratio is a model feature")
    add("Lat/lon not true city coordinates (e.g. Los Angeles lat ~28.6)", "yes", "yes",
        "Used as consistent per-city coordinates + haversine, not as literal geography")
    unseen = (set(val.pickup) | set(val.delivery)) - (set(train.pickup) | set(train.delivery))
    add("Cities in validation never seen in training", "-", f"{len(unseen)}: {', '.join(sorted(unseen))}",
        "Geography/distance features (not city IDs) so new cities still get sensible predictions")
    lanes_tr = set(zip(train.pickup, train.delivery))
    lanes_va = set(zip(val.pickup, val.delivery))
    add("Validation lanes unseen in training", "-", len(lanes_va - lanes_tr), "Same as above; no lane ID features")
    add("market_index mean (distribution shift)", round(train.market_index.mean(), 3),
        round(val.market_index.mean(), 3), "Level not used; drift would extrapolate badly")
    add("Duplicate load_id", int(train.load_id.duplicated().sum()), int(val.load_id.duplicated().sum()), "None found")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(err) / y) * 100),
        "WAPE_%": float(np.sum(np.abs(err)) / np.sum(y) * 100),
        "MedAPE_%": float(np.median(np.abs(err) / y) * 100),
    }


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def fit_lgbm(X: pd.DataFrame, y: pd.Series, seeds=SEEDS, params=None):
    models = []
    for s in seeds:
        p = dict(LGB_PARAMS)
        p.update(params or {})
        p["random_state"] = s
        models.append(lgb.LGBMRegressor(**p).fit(X, y, categorical_feature=["equip"]))
    return models


def predict_rate(models, X: pd.DataFrame, distance: np.ndarray) -> np.ndarray:
    log_rpm = np.mean([m.predict(X) for m in models], axis=0)
    return np.exp(log_rpm) * distance


def market_features(df: pd.DataFrame, daily_mi: pd.Series) -> pd.DataFrame:
    """Features used ONLY in the rejected ablation variant."""
    f = pd.DataFrame(index=df.index)
    f["mi_day"] = df["date"].map(daily_mi)
    f["mi_row"] = df["market_index"].fillna(f["mi_day"])
    f["quote_signal"] = df["quote_signal"]
    return f


# --------------------------------------------------------------------------------------
# Temporal cross-validation (+ ablation)
# --------------------------------------------------------------------------------------
def rolling_cv(train: pd.DataFrame, val: pd.DataFrame, fast: bool = False):
    daily_mi = pd.concat([train[["date", "market_index"]], val[["date", "market_index"]]]) \
        .groupby("date").market_index.median()
    variants = ["Baseline: equipment x distance-bucket median rate/mile",
                "LightGBM + market_index + quote_signal (rejected)",
                "LightGBM, trained on uncleaned labels",
                "LightGBM FINAL (stable features, cleaned labels)"]
    rows = []
    for k, (a, b, c, e) in enumerate(CV_FOLDS, start=1):
        A = train[(train.date >= a) & (train.date <= b)]
        B = train[(train.date >= c) & (train.date <= e)]
        wmed = A.weight.abs().median()
        bad_A = corrupted_label_mask(A)
        clean_B = ~corrupted_label_mask(B)
        y_log = np.log(A.posted_rate / A.distance)
        XA, XB = build_features(A, wmed), build_features(B, wmed)
        preds = {}

        # 1) baseline
        med = (A.posted_rate / A.distance).groupby([A.equipment, pd.cut(A.distance, DIST_BINS)], observed=True).median()
        key = pd.MultiIndex.from_arrays([B.equipment, pd.cut(B.distance, DIST_BINS)])
        preds[variants[0]] = med.reindex(key).values * B.distance.values

        seeds = (0,) if fast else (0, 1, 2)
        # 2) + market_index + quote_signal (rejected)
        XA2 = pd.concat([XA, market_features(A, daily_mi)], axis=1)
        XB2 = pd.concat([XB, market_features(B, daily_mi)], axis=1)
        m2 = fit_lgbm(XA2[~bad_A], y_log[~bad_A], seeds)
        preds[variants[1]] = predict_rate(m2, XB2, B.distance.values)
        # 3) no label cleaning
        m3 = fit_lgbm(XA, y_log, seeds)
        preds[variants[2]] = predict_rate(m3, XB, B.distance.values)
        # 4) final
        m4 = fit_lgbm(XA[~bad_A], y_log[~bad_A], seeds)
        preds[variants[3]] = predict_rate(m4, XB, B.distance.values)

        for name, p in preds.items():
            for scope, mask in (("all rows", np.ones(len(B), bool)), ("clean rows", clean_B.values)):
                rows.append({"fold": k, "train": f"{a[:7]}..{b[:7]}", "test": f"{c[:7]}..{e[:7]}",
                             "model": name, "scope": scope,
                             **metrics(B.posted_rate.values[mask], np.asarray(p)[mask])})
        print(f"  fold {k}: train {a[:7]}..{b[:7]} -> test {c[:7]}..{e[:7]}  (n_test={len(B):,})")
    res = pd.DataFrame(rows)
    summary = res.groupby(["model", "scope"], sort=False).mean(numeric_only=True).drop(columns="fold").round(3).reset_index()
    return res, summary


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
TEAL, GREY, RED = "#064A56", "#9DAFB3", "#C0392B"


def fig_split(outdir: Path):
    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=150)
    months = pd.period_range("2025-01", "2025-12", freq="M")
    ax.set_xlim(0, 12); ax.set_ylim(-0.5, 4.5)
    rows = [("Fold 1", 0, 6, 6, 8), ("Fold 2", 0, 7, 7, 9), ("Fold 3", 0, 8, 8, 10)]
    for i, (name, t0, t1, v0, v1) in enumerate(rows):
        y = 3 - i
        ax.barh(y, t1 - t0, left=t0, color=TEAL, height=0.55)
        ax.barh(y, v1 - v0, left=v0, color="#E0A526", height=0.55)
        ax.text(-0.1, y, name, ha="right", va="center", fontsize=9)
    ax.barh(-0.0, 10, left=0, color=TEAL, height=0.55)
    ax.barh(-0.0, 2, left=10, color=RED, height=0.55)
    ax.text(-0.1, 0, "Final", ha="right", va="center", fontsize=9)
    ax.set_xticks(np.arange(12) + 0.5); ax.set_xticklabels([m.strftime("%b") for m in months], fontsize=8)
    ax.set_yticks([])
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.set_title("Time-based validation: always train on the past, test on the future", loc="left", fontsize=11, fontweight="bold")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TEAL), plt.Rectangle((0, 0), 1, 1, color="#E0A526"),
                       plt.Rectangle((0, 0), 1, 1, color=RED)],
              labels=["Train", "CV test (labels known)", "Final validation Nov-Dec (no labels)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "fig_split_scheme.png", bbox_inches="tight"); plt.close(fig)


def fig_cv(summary: pd.DataFrame, outdir: Path):
    s = summary[summary.scope == "clean rows"].copy()
    short = {s_: s_.replace("LightGBM ", "LGBM ").replace("Baseline: equipment x distance-bucket median rate/mile", "Baseline (bucket median)")
             for s_ in s.model}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150)
    colors = [GREY, RED, "#E0A526", TEAL]
    for ax, col, title in zip(axes, ["MAE", "MAPE_%"], ["MAE ($)  - lower is better", "MAPE (%)  - lower is better"]):
        ax.barh([short[m] for m in s.model][::-1], s[col].values[::-1], color=colors[::-1])
        for i, v in enumerate(s[col].values[::-1]):
            ax.text(v, i, f" {v:.1f}" if col == "MAE" else f" {v:.2f}", va="center", fontsize=8)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[1].set_yticklabels([])
    fig.suptitle("Rolling-origin CV (mean of 3 time folds, clean-label rows)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(outdir / "fig_cv_metrics.png", bbox_inches="tight"); plt.close(fig)


def signal_instability_metrics(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    t = train.copy()
    t["dev"] = label_deviation(t)
    c = t[t.dev.abs() <= LABEL_DEV_THRESHOLD].copy()
    c["month"] = c.date.dt.to_period("M")
    qs_corr = c.groupby("month").apply(lambda g: g.quote_signal.corr(g.dev))
    allm = pd.concat([train[["date", "market_index"]], val[["date", "market_index"]]])
    allm["month"] = allm.date.dt.to_period("M")
    mi_month = allm.groupby("month").market_index.mean()
    return {"quote_signal_monthly_corr": {str(k): round(float(v), 3) for k, v in qs_corr.items()},
            "market_index_monthly_mean": {str(k): round(float(v), 3) for k, v in mi_month.items()}}


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Train, validate and predict for the Freight Rate challenge.")
    ap.add_argument("--data-dir", default="data", help="folder with the CSV files (default: ./data, falls back to ./)")
    ap.add_argument("--out-dir", default=str(here / "outputs"), help="folder for tables/figures/summary")
    ap.add_argument("--predictions-out", default=str(here / "validation_predictions.csv"))
    ap.add_argument("--skip-cv", action="store_true", help="skip temporal CV + ablation (faster)")
    ap.add_argument("--fast", action="store_true", help="single seed inside CV (faster; final model still 3 seeds)")
    ap.add_argument("--skip-scorer", action="store_true", help="do not call score.py at the end")
    args = ap.parse_args()

    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    print("== 1/6 Loading data")
    train, val, template, dec, paths = load_inputs(Path(args.data_dir))
    for k, v in paths.items():
        print(f"   {k:9s}: {v}")
    print(f"   train/test rows={len(train):,} | validation rows={len(val):,} | december rows={len(dec)}")

    print("== 2/6 Data-quality audit")
    audit = audit_data(train, val)
    audit.to_csv(outdir / "data_quality_report.csv", index=False)
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        print(audit[["issue", "train_test", "validation"]].to_string(index=False))

    print("== 3/6 Evidence figures")
    fig_split(outdir)
    instability = signal_instability_metrics(train, val)

    summary_cv, cv_table = None, None
    if not args.skip_cv:
        print("== 4/6 Temporal cross-validation (rolling-origin) + ablation")
        cv_full, cv_table = rolling_cv(train, val, fast=args.fast)
        cv_full.to_csv(outdir / "cv_results_by_fold.csv", index=False)
        cv_table.to_csv(outdir / "cv_results.csv", index=False)
        fig_cv(cv_table, outdir)
        print(cv_table.to_string(index=False))
        summary_cv = cv_table.to_dict(orient="records")
    else:
        print("== 4/6 CV skipped")

    print("== 5/6 Final fit on ALL train/test data and predict validation + December")
    wmed = float(train.weight.abs().median())
    bad = corrupted_label_mask(train)
    y_log = np.log(train.posted_rate / train.distance)
    X_train = build_features(train, wmed)
    models = fit_lgbm(X_train[~bad], y_log[~bad])
    print(f"   trained on {int((~bad).sum()):,} rows ({int(bad.sum())} corrupted labels excluded), {len(models)} seeds")
    # --- validation predictions (aligned to the template's load_id order) ---
    X_val = build_features(val, wmed)
    val_pred = predict_rate(models, X_val, val.distance.values)
    pred_df = pd.DataFrame({"load_id": val.load_id.values, "predicted_rate": np.round(val_pred, 2)})
    out = template[["load_id"]].merge(pred_df, on="load_id", how="left")
    assert len(out) == EXPECTED_ROWS and out.predicted_rate.notna().all() and (out.predicted_rate > 0).all(), \
        "validation predictions failed sanity checks"
    out.to_csv(args.predictions_out, index=False)
    print(f"   wrote {args.predictions_out}  (rows={len(out):,}, mean=${out.predicted_rate.mean():,.0f}, "
          f"min=${out.predicted_rate.min():,.0f}, max=${out.predicted_rate.max():,.0f})")

    # --- December chart predictions: only the date changes ---
    coords = {}
    for p, la, lo in (("pickup", "pickup_lat", "pickup_lon"), ("delivery", "delivery_lat", "delivery_lon")):
        for city, grp in pd.concat([train, val]).groupby(p):
            coords[city] = (grp[la].iloc[0], grp[lo].iloc[0])
    d2 = dec.copy()
    d2["date"] = pd.to_datetime(d2["date"])
    d2["pickup_lat"] = d2.pickup.map(lambda c: coords[c][0]); d2["pickup_lon"] = d2.pickup.map(lambda c: coords[c][1])
    d2["delivery_lat"] = d2.delivery.map(lambda c: coords[c][0]); d2["delivery_lon"] = d2.delivery.map(lambda c: coords[c][1])
    d2["weight"] = d2["weight"].astype(float)
    dec_pred = predict_rate(models, build_features(d2, wmed), d2.distance.astype(float).values)
    dec_out = dec.copy()
    dec_out["predicted_rate"] = np.round(dec_pred, 2)
    dec_path = outdir / "december-chart-inputs.csv"
    dec_out.to_csv(dec_path, index=False)
    print(f"   wrote {dec_path}  (min=${dec_pred.min():,.2f}, max=${dec_pred.max():,.2f}; only the date changes)")

    print("== 6/6 Running the provided scorer (validates files + builds candidate_december.png)")
    scorer_ok = None
    if not args.skip_scorer:
        score = next((p for p in [Path("score.py"), here / "score.py", Path(args.data_dir).parent / "score.py"] if p.is_file()), None)
        if score is None:
            print("   score.py not found next to this script - copy it here and run:")
            print(f"   python score.py --predictions {args.predictions_out} --december-predictions {dec_path}")
        else:
            r = subprocess.run([sys.executable, str(score), "--predictions", args.predictions_out,
                                "--december-predictions", str(dec_path)], capture_output=True, text=True)
            print("   " + (r.stdout + r.stderr).strip().replace("\n", "\n   "))
            scorer_ok = r.returncode == 0

    summary = {
        "train_rows": int(len(train)), "validation_rows": int(len(val)),
        "corrupted_labels_removed_from_training": int(bad.sum()),
        "model": "LightGBM on log(rate/mile), 3-seed average", "lgb_params": {k: v for k, v in LGB_PARAMS.items()},
        "features": STABLE_FEATURES, "excluded_features": ["market_index", "quote_signal"],
        "cv_folds": CV_FOLDS, "cv_summary": summary_cv, "signal_instability": instability,
        "december_pred_min": float(dec_pred.min()), "december_pred_max": float(dec_pred.max()),
        "validation_pred_mean": float(out.predicted_rate.mean()), "scorer_passed": scorer_ok,
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if scorer_ok is False:
        raise SystemExit("Scorer failed; candidate_december.png was not created.")
    print("\nDone. Deliverables: validation_predictions.csv | scorer_results/candidate_december.png | outputs/*")


if __name__ == "__main__":
    main()
