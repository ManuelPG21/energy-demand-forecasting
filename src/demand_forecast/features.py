"""Calendar and lag features. Colombia moves most holidays to Monday ("puentes"), which strongly lowers demand."""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    co = holidays.Colombia(years=range(index.year.min(), index.year.max() + 2))
    hol = pd.Series(index.map(lambda d: d in co), index=index).astype(int)
    df = pd.DataFrame(index=index)
    df["dow"] = index.dayofweek
    df["month"] = index.month
    df["doy_sin"] = np.sin(2 * np.pi * index.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * index.dayofyear / 365.25)
    df["holiday"] = hol
    df["pre_holiday"] = hol.shift(-1, fill_value=0)
    df["post_holiday"] = hol.shift(1, fill_value=0)
    df["dec_holidays"] = ((index.month == 12) & (index.day >= 20) | (index.month == 1) & (index.day <= 6)).astype(int)
    df["trend"] = (index - pd.Timestamp("2019-01-01")).days / 365.25
    return df


def direct_features(y: pd.Series, horizon: int) -> pd.DataFrame:
    """Lag features usable for a direct forecast `horizon` days ahead (only lags >= horizon)."""
    df = calendar_features(y.index)
    for lag in (horizon, horizon + 7, horizon + 14, 364):
        df[f"lag_{lag}"] = y.shift(lag)
    base = y.shift(horizon)
    df[f"roll7_mean_h{horizon}"] = base.rolling(7).mean()
    df[f"roll28_mean_h{horizon}"] = base.rolling(28).mean()
    return df
