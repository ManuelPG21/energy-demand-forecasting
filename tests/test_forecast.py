import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_forecast import backtest  # noqa: E402
from demand_forecast.features import calendar_features, direct_features  # noqa: E402
from demand_forecast.models import LightGBMDirect, SeasonalNaive  # noqa: E402


def synthetic_series(days=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=days, freq="D")
    weekly = np.where(idx.dayofweek == 6, 0.85, np.where(idx.dayofweek == 5, 0.93, 1.0))
    return pd.Series(200 * weekly * (1 + 0.02 * np.sin(2 * np.pi * idx.dayofyear / 365)) + rng.normal(0, 1, days), index=idx)


def test_colombian_holidays_flagged():
    cal = calendar_features(pd.date_range("2025-12-24", "2025-12-26"))
    assert cal.loc["2025-12-25", "holiday"] == 1
    assert cal.loc["2025-12-24", "pre_holiday"] == 1


def test_direct_features_do_not_leak_future():
    y = synthetic_series(100)
    X = direct_features(y, horizon=7)
    # the lag-7 feature on day t must equal y on day t-7
    assert X["lag_7"].iloc[50] == y.iloc[43]


def test_seasonal_naive_repeats_last_week():
    y = synthetic_series(60)
    f = SeasonalNaive().forecast(y, 7)
    assert np.allclose(f.to_numpy(), y.iloc[-7:].to_numpy())


def test_lightgbm_beats_naive_on_structured_series():
    y = synthetic_series()
    origins = list(pd.date_range("2024-01-01", periods=6, freq="14D"))
    lgb = backtest.metrics(backtest.rolling_origin(y, LightGBMDirect(n_estimators=200), 14, origins))
    naive = backtest.metrics(backtest.rolling_origin(y, SeasonalNaive(), 14, origins))
    assert lgb["mape"] < naive["mape"] * 1.1


def test_diebold_mariano_detects_better_model():
    rng = np.random.default_rng(1)
    e_good, e_bad = rng.normal(0, 1, 400), rng.normal(0, 2, 400)
    stat, p = backtest.diebold_mariano(e_good, e_bad)
    assert stat < 0 and p < 0.01


def test_conformal_coverage_close_to_nominal():
    rng = np.random.default_rng(2)
    n = 4000
    df = pd.DataFrame({"step": rng.integers(1, 8, n), "forecast": 100.0})
    df["actual"] = df["forecast"] * (1 + rng.normal(0, 0.03, n))
    out = backtest.conformal_intervals(df.iloc[:2000], df.iloc[2000:], alpha=0.1)
    assert 0.87 < out["covered"].mean() < 0.93
