#!/usr/bin/env python3
"""
Daily runner for fetching pools, updating the cache, and running the top-N backtest.
Suitable for long-running use on a Raspberry Pi (or any host).
"""
from __future__ import annotations

import time
from pathlib import Path

from backtest import run_backtest, summarize
from cache_utils import ensure_today_snapshot


def run_daily() -> None:
    """Fetch today's snapshot, run backtest over recent days, and emit a tail printout."""
    ensure_today_snapshot()
    eq = run_backtest(
        top_n=10,
        days=30,
        il_mode="heuristic",
        min_tvl=1_000_000,
        csv_path=Path("equity.csv"),
    )
    if not eq.empty:
        summary = summarize(eq)
        if summary:
            print("Summary:", summary)
        print(eq.tail())
    else:
        print("No data to backtest.")


def main() -> int:
    # Run once per day; sleep between runs.
    while True:
        run_daily()
        time.sleep(24 * 60 * 60)


if __name__ == "__main__":
    raise SystemExit(main())
