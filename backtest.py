#!/usr/bin/env python3
"""Simple top-N rotation backtest for DeFiLlama pools.

- Pulls daily snapshots from local cache (data/pools_YYYY-MM-DD.json)
- Falls back to fetching today's snapshot if missing
- Ranks pools with ai_yield_tools score_pools()
- Allocates equally into top-N each day and compounds with daily APY
- Optional IL modeling: apply il7d penalty or a simple volatility-based haircut

Outputs summary stats and a CSV of the equity curve.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
import typing as t

import numpy as np
import pandas as pd

from ai_yield_tools import score_pools, top_safe_apys
from cache_utils import ensure_today_snapshot, list_snapshots, load_snapshot


def apy_to_daily_rate(apy: float) -> float:
    if apy is None or math.isnan(apy):
        return 0.0
    return (1 + apy / 100) ** (1 / 365) - 1


def apply_il_penalty(row: pd.Series, mode: str = "none") -> float:
    """Return IL drag as daily return adjustment (negative value)."""
    if mode == "none":
        return 0.0
    il7d = row.get("il7d")
    vol_proxy = row.get("token_volatility") if "token_volatility" in row else 0
    if mode == "il7d" and pd.notna(il7d):
        return float(il7d) / 7 / 100  # convert percent per day
    # simple heuristic: tie IL drag to volatility proxy
    return min(vol_proxy, 5) * -0.0005  # -5 bps per unit vol capped


def run_backtest(top_n: int, days: int, il_mode: str, min_tvl: float, csv_path: Path | None) -> pd.DataFrame:
    snaps = list_snapshots()
    if not snaps:
        ensure_today_snapshot()
        snaps = list_snapshots()
    snaps = snaps[-days:]  # take the latest N snapshots available

    equity = []
    value = 1.0  # start at 1 unit capital
    for snap_path in snaps:
        df = load_snapshot(snap_path)
        scored = score_pools(df)
        if min_tvl > 0:
            scored = scored[scored["tvlUsd"] >= min_tvl]
        ranked = top_safe_apys(scored, n=top_n)
        ranked = ranked.merge(scored[["pool", "token_volatility", "il7d"]], left_index=True, right_index=True, how="left") if "pool" in scored.columns else ranked

        daily_ret = 0.0
        picks = ranked.head(top_n)
        if len(picks) == 0:
            equity.append({"date": snap_path.stem, "value": value})
            continue
        weight = 1 / len(picks)
        for _, row in picks.iterrows():
            dr = apy_to_daily_rate(row.get("apy", 0))
            il_drag = apply_il_penalty(row, mode=il_mode)
            daily_ret += weight * (dr + il_drag)
        value *= (1 + daily_ret)
        equity.append({"date": snap_path.stem, "value": value, "daily_ret": daily_ret, "picks": len(picks)})

    equity_df = pd.DataFrame(equity)
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        equity_df.to_csv(csv_path, index=False)
    return equity_df


def summarize(equity: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {}
    returns = equity["daily_ret"].fillna(0)
    cumulative = equity["value"].iloc[-1] - 1
    max_drawdown = ((equity["value"].cummax() - equity["value"]) / equity["value"].cummax()).max()
    vol = returns.std() * math.sqrt(365)
    return {
        "days": len(equity),
        "cumulative_return": cumulative,
        "max_drawdown": max_drawdown,
        "annualized_vol": vol,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top", type=int, default=10, help="number of pools to hold each day (default 10)")
    p.add_argument("--days", type=int, default=30, help="number of most recent snapshots to backtest")
    p.add_argument("--il", choices=["none", "il7d", "heuristic"], default="none", help="apply IL penalty mode")
    p.add_argument("--min-tvl", type=float, default=0, help="minimum TVL filter (USD)")
    p.add_argument("--csv", type=Path, help="optional path to write equity curve CSV")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    eq = run_backtest(args.top, args.days, args.il, args.min_tvl, args.csv)
    summary = summarize(eq)
    if summary:
        print("Summary:", summary)
    else:
        print("No data to backtest.")
    if not eq.empty:
        print(eq.tail())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
