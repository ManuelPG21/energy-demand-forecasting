"""Two-year rolling-origin backtest of 14-day-ahead national demand forecasts, logged to MLflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demand_forecast import backtest  # noqa: E402
from demand_forecast.data import load_daily_demand  # noqa: E402
from demand_forecast.models import LightGBMDirect, SarimaxHolidays, SeasonalNaive  # noqa: E402

REPORTS = ROOT / "reports"
HORIZON = 14


def main() -> None:
    y = load_daily_demand()
    origins = list(pd.date_range("2024-09-01", y.index[-1] - pd.Timedelta(days=HORIZON), freq=f"{HORIZON}D"))
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("co-demand-forecast")

    frames, summary = [], {}
    for model in (SeasonalNaive(), SarimaxHolidays(), LightGBMDirect()):
        bt = backtest.rolling_origin(y, model, HORIZON, origins)
        frames.append(bt)
        summary[model.name] = backtest.metrics(bt)
        with mlflow.start_run(run_name=model.name):
            mlflow.log_params({"model": model.name, "horizon": HORIZON, "n_origins": len(origins),
                               "first_origin": str(origins[0].date()), **getattr(model, "params", {})})
            mlflow.log_metrics(summary[model.name])
        print(model.name, summary[model.name])

    # Equal-weight ensemble of the two structurally different models (no extra fitting).
    s, g = frames[1].set_index(["origin", "date"]), frames[2].set_index(["origin", "date"])
    ens = s.assign(forecast=(s["forecast"] + g.loc[s.index, "forecast"]) / 2, model="ensemble").reset_index()
    frames.append(ens)
    summary["ensemble"] = backtest.metrics(ens)
    with mlflow.start_run(run_name="ensemble"):
        mlflow.log_params({"model": "ensemble", "members": "sarimax+lightgbm", "horizon": HORIZON, "n_origins": len(origins)})
        mlflow.log_metrics(summary["ensemble"])
    print("ensemble", summary["ensemble"])

    all_bt = pd.concat(frames, ignore_index=True)
    all_bt.to_csv(REPORTS / "backtest_forecasts.csv", index=False)
    wide = all_bt.pivot_table(index=["origin", "date"], values="forecast", columns="model").join(
        all_bt.drop_duplicates(["origin", "date"]).set_index(["origin", "date"])["actual"])
    err = {m: (wide[m] - wide["actual"]).to_numpy() for m in summary}

    pairs = [("lightgbm", "sarimax"), ("lightgbm", "seasonal_naive"), ("ensemble", "sarimax"), ("ensemble", "lightgbm")]
    dm = {f"{a}_vs_{b}": dict(zip(("stat", "p_value"), backtest.diebold_mariano(err[a], err[b], h=HORIZON))) for a, b in pairs}

    # Conformal 90% intervals for the best model: calibrate on the first year of folds, test on the second.
    best = min(summary, key=lambda m: summary[m]["mape"])
    b = all_bt[all_bt["model"] == best]
    split = origins[len(origins) // 2]
    ci = backtest.conformal_intervals(b[b["origin"] < split], b[b["origin"] >= split], alpha=0.10)
    coverage = float(ci["covered"].mean())

    by_step = all_bt.assign(ape=(all_bt["forecast"] - all_bt["actual"]).abs() / all_bt["actual"] * 100) \
        .groupby(["model", "step"])["ape"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    by_step.plot(ax=ax, marker="o")
    ax.set(xlabel="days ahead", ylabel="MAPE [%]", title="Forecast error by horizon (2-year backtest)")
    fig.tight_layout(); fig.savefig(REPORTS / "figures" / "mape_by_horizon.png", dpi=150); plt.close(fig)

    last = ci[ci["origin"] >= ci["origin"].max() - pd.Timedelta(days=8 * HORIZON)]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(y.loc[last["date"].min():last["date"].max()], "k-", lw=1.2, label="actual")
    for _, g in last.groupby("origin"):
        ax.plot(g["date"], g["forecast"], c="tab:blue", lw=1)
        ax.fill_between(g["date"], g["lower"], g["upper"], color="tab:blue", alpha=0.15)
    ax.set(ylabel="GWh / day", title=f"{best}: 14-day forecasts with 90% conformal intervals (last 16 weeks)")
    ax.plot([], [], c="tab:blue", label="forecast + 90% interval"); ax.legend(loc="lower left")
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(REPORTS / "figures" / "forecasts.png", dpi=150); plt.close(fig)

    report = {"series": {"start": str(y.index[0].date()), "end": str(y.index[-1].date()), "days": len(y)},
              "backtest": {"horizon_days": HORIZON, "origins": len(origins), "first_origin": str(origins[0].date())},
              "metrics": summary, "diebold_mariano": dm, "best_model": best,
              "conformal_90": {"nominal": 0.90, "empirical_coverage": coverage}}
    (REPORTS / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
