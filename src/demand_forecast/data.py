"""Download hourly national demand from the XM public API (Colombian grid operator) and build a daily series."""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

XM_URL = "https://servapibi.xm.com.co/hourly"
DATA = Path(__file__).resolve().parents[2] / "data"


def _fetch_chunk(start: date, end: date) -> pd.DataFrame:
    body = {"MetricId": "DemaReal", "StartDate": start.isoformat(), "EndDate": end.isoformat(), "Entity": "Sistema", "Filter": []}
    for attempt in range(5):
        try:
            r = requests.post(XM_URL, json=body, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    rows = []
    for item in r.json()["Items"]:
        for ent in item["HourlyEntities"]:
            vals = ent["Values"]
            hours = [float(vals[f"Hour{h:02d}"]) for h in range(1, 25) if vals.get(f"Hour{h:02d}") not in (None, "")]
            rows.append({"date": item["Date"], "demand_gwh": sum(hours) / 1e6, "hours": len(hours)})
    return pd.DataFrame(rows)


def load_daily_demand(start: str = "2019-01-01", end: str = "2026-08-31", refresh: bool = False) -> pd.Series:
    """Daily national electricity demand in GWh (sum of 24 hourly values, reported in kWh)."""
    path = DATA / "daily_demand.csv"
    if path.exists() and not refresh:
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        DATA.mkdir(exist_ok=True)
        chunks, cur, stop = [], date.fromisoformat(start), date.fromisoformat(end)
        while cur <= stop:
            nxt = min(cur + timedelta(days=29), stop)
            chunks.append(_fetch_chunk(cur, nxt))
            cur = nxt + timedelta(days=1)
            time.sleep(0.5)
        df = pd.concat(chunks, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["hours"] == 24].drop(columns="hours").drop_duplicates("date").sort_values("date")
        df.to_csv(path, index=False)
    return df.set_index("date")["demand_gwh"].asfreq("D")
