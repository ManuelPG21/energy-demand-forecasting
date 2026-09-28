"""Rolling-origin backtesting, error metrics, Diebold-Mariano test and conformal prediction intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rolling_origin(y: pd.Series, model, horizon: int, origins: list[pd.Timestamp]) -> pd.DataFrame:
    """Refit at each origin using data up to that day only, forecast the next `horizon` days."""
    rows = []
    for origin in origins:
        history = y.loc[:origin]
        pred = model.forecast(history, horizon)
        actual = y.reindex(pred.index)
        rows.append(pd.DataFrame({"origin": origin, "date": pred.index, "step": np.arange(1, len(pred) + 1),
                                  "forecast": pred.to_numpy(), "actual": actual.to_numpy(), "model": model.name}))
    return pd.concat(rows, ignore_index=True).dropna(subset=["actual"])


def metrics(df: pd.DataFrame) -> dict:
    err = df["forecast"] - df["actual"]
    return {"mape": float(np.mean(np.abs(err) / df["actual"]) * 100), "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))), "bias": float(np.mean(err))}


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """DM test on absolute-error loss differentials with a Newey-West variance (lag h-1).

    Negative statistic: model 1 is more accurate. Returns (statistic, two-sided p-value).
    """
    d = np.abs(e1) - np.abs(e2)
    n, mean = len(d), d.mean()
    gamma = [np.sum((d[k:] - mean) * (d[: n - k] - mean)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    stat = mean / np.sqrt(var)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def conformal_intervals(calib: pd.DataFrame, test: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    """Split-conformal intervals per horizon step from relative errors observed in earlier backtest folds."""
    rel = (calib["actual"] - calib["forecast"]) / calib["forecast"]
    q = rel.abs().groupby(calib["step"]).quantile(1 - alpha)
    out = test.copy()
    width = out["step"].map(q).fillna(q.max()) * out["forecast"]
    out["lower"], out["upper"] = out["forecast"] - width, out["forecast"] + width
    out["covered"] = (out["actual"] >= out["lower"]) & (out["actual"] <= out["upper"])
    return out
