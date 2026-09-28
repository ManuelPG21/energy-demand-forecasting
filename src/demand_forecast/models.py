"""Forecasters with a common interface: fit on history, predict the next `horizon` days."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .features import calendar_features, direct_features


class SeasonalNaive:
    """Same weekday last week. The baseline every model must beat."""

    name = "seasonal_naive"

    def forecast(self, history: pd.Series, horizon: int) -> pd.Series:
        idx = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.Series([history.iloc[-7 + (h % 7)] for h in range(horizon)], index=idx)


class SarimaxHolidays:
    """SARIMA(1,0,1)x(1,1,1,7) on log demand with holiday / calendar regressors."""

    name = "sarimax"
    exog_cols = ["holiday", "pre_holiday", "post_holiday", "dec_holidays", "doy_sin", "doy_cos"]

    def __init__(self, train_window: int = 3 * 365):
        self.train_window = train_window

    def forecast(self, history: pd.Series, horizon: int) -> pd.Series:
        h = history.iloc[-self.train_window:]
        idx = pd.date_range(h.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        exog = calendar_features(h.index.append(idx))[self.exog_cols]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(np.log(h), exog=exog.loc[h.index], order=(1, 0, 1), seasonal_order=(1, 1, 1, 7),
                          trend="c").fit(disp=False, maxiter=200)
            pred = fit.forecast(horizon, exog=exog.loc[idx])
        return pd.Series(np.exp(pred.to_numpy()), index=idx)


class LightGBMDirect:
    """Gradient boosting, one model per horizon step group (direct strategy, no recursive error build-up).

    Trained on log demand; lags respect the horizon so no future information leaks into features.
    """

    name = "lightgbm"

    def __init__(self, groups=((1, 7), (8, 14)), **params):
        self.groups = groups
        self.params = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1) | params

    def forecast(self, history: pd.Series, horizon: int) -> pd.Series:
        idx = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        y_ext = pd.concat([history, pd.Series(np.nan, index=idx)])
        log_y = np.log(y_ext)
        preds = {}
        for lo, hi in self.groups:
            if lo > horizon:
                break
            X = direct_features(log_y, hi)
            train = X.loc[history.index].assign(y=log_y.loc[history.index]).dropna()
            model = LGBMRegressor(**self.params).fit(train.drop(columns="y"), train["y"])
            steps = idx[lo - 1:min(hi, horizon)]
            preds.update(zip(steps, np.exp(model.predict(X.loc[steps]))))
        return pd.Series(preds).sort_index()
