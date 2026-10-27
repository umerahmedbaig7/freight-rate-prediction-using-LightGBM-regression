import numpy as np
import pandas as pd
import pytest

import freight_rate_pipeline as frp


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "distance": rng.uniform(100, 1500, n),
        "equipment": rng.choice(list(frp.EQUIP_MAP), n),
        "weight": rng.uniform(10_000, 47_500, n),
        "pickup_lat": rng.uniform(30, 45, n),
        "pickup_lon": rng.uniform(-100, -75, n),
        "delivery_lat": rng.uniform(30, 45, n),
        "delivery_lon": rng.uniform(-100, -75, n),
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
    })
    df.loc[0, "distance"] = 1000.0
    df.loc[0, "equipment"] = "Dry Van"
    df["posted_rate"] = df["distance"] * 2.0  # constant $2/mile
    return df


def test_haversine_one_degree_longitude_at_equator():
    df = pd.DataFrame({"pickup_lat": [0.0], "pickup_lon": [0.0],
                       "delivery_lat": [0.0], "delivery_lon": [1.0]})
    assert frp.haversine_miles(df)[0] == pytest.approx(69.09, abs=0.1)


def test_build_features_cleans_weight_and_keeps_schema():
    df = _toy()
    df.loc[0, "weight"] = -20_000.0
    df.loc[1, "weight"] = np.nan
    X = frp.build_features(df, weight_median=30_000.0)
    assert list(X.columns) == frp.STABLE_FEATURES
    assert X.loc[0, "weight"] == 20_000.0
    assert X.loc[1, "weight"] == 30_000.0
    assert X["weight"].notna().all()


def test_corrupted_label_detector_flags_injected_spike():
    df = _toy()
    df.loc[0, "posted_rate"] *= 3  # 3x rate spike
    mask = frp.corrupted_label_mask(df)
    assert bool(mask.iloc[0]) and int(mask.sum()) == 1


def test_fit_and_predict_returns_positive_rates():
    df = _toy()
    X = frp.build_features(df, weight_median=30_000.0)
    y = np.log(df.posted_rate / df.distance)
    models = frp.fit_lgbm(X, y, seeds=(0,),
                          params={"n_estimators": 20, "min_child_samples": 10})
    pred = frp.predict_rate(models, X, df.distance.values)
    assert (pred > 0).all()
    assert np.allclose(pred, df.distance.values * 2.0, rtol=0.05)
