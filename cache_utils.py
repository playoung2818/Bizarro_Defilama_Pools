"""Caching helper for DeFiLlama pools daily snapshots."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import typing as t

import pandas as pd

from ai_yield_tools import FetchError, fetch_pools_df

CACHE_DIR = Path("data")
CACHE_DIR.mkdir(exist_ok=True)


def snapshot_filename(ts: datetime | None = None) -> Path:
    ts = ts or datetime.now(timezone.utc)
    return CACHE_DIR / f"pools_{ts.strftime('%Y-%m-%d')}.json"


def save_snapshot(df: pd.DataFrame, ts: datetime | None = None) -> Path:
    path = snapshot_filename(ts)
    payload = {
        "timestamp": (ts or datetime.now(timezone.utc)).isoformat(),
        "data": df.to_dict(orient="records"),
    }
    path.write_text(json.dumps(payload))
    return path


def load_snapshot(path: Path) -> pd.DataFrame:
    obj = json.loads(path.read_text())
    data = obj.get("data", [])
    return pd.DataFrame(data)


def list_snapshots() -> list[Path]:
    return sorted(CACHE_DIR.glob("pools_*.json"))


def ensure_today_snapshot() -> Path:
    path = snapshot_filename()
    if path.exists():
        return path
    df = fetch_pools_df()
    save_snapshot(df, datetime.now(timezone.utc))
    return path


def fetch_and_cache(ts: datetime | None = None) -> Path:
    df = fetch_pools_df()
    return save_snapshot(df, ts)


__all__ = [
    "CACHE_DIR",
    "snapshot_filename",
    "save_snapshot",
    "load_snapshot",
    "list_snapshots",
    "ensure_today_snapshot",
    "fetch_and_cache",
]
