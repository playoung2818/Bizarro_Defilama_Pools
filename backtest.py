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

from ai_yield_tools import DEFAULT_CHAIN, filter_chain, score_pools, tag_pools, top_safe_apys
from cache_utils import ensure_today_snapshot, list_snapshots, load_snapshot
from logistic_risk_model import get_logistic_risk_model, score_pools_logistic
from xgb_scoring import XGBTrainingError, XGBUnavailableError, score_pools_xgb, train_xgb_from_cache


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


def run_backtest(
    top_n: int,
    days: int,
    il_mode: str,
    min_tvl: float,
    csv_path: Path | None,
    include_tags: list[str] | None = None,
    model_type: str = "heuristic",
    retrain: bool = False,
) -> pd.DataFrame:
    snaps = list_snapshots()
    if not snaps:
        ensure_today_snapshot()
        snaps = list_snapshots()
    snaps = snaps[-days:]  # take the latest N snapshots available
    logit_model = None
    xgb_model = None
    if model_type == "logit":
        logit_model, _ = get_logistic_risk_model(retrain=retrain, max_pairs=max(days, 30))
    if model_type == "xgb":
        xgb_model = train_xgb_from_cache(max_pairs=max(days, 30))

    equity = []
    value = 1.0  # start at 1 unit capital
    for snap_path in snaps:
        df = load_snapshot(snap_path)
        df = filter_chain(df)
        if model_type == "logit" and logit_model is not None:
            scored = score_pools_logistic(df, logit_model)
        elif model_type == "xgb" and xgb_model is not None:
            scored = score_pools_xgb(df, xgb_model)
        else:
            scored = score_pools(df)
        if include_tags:
            tagged = tag_pools(scored)
            mask = tagged[include_tags].any(axis=1)
            scored = tagged.loc[mask]
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
    mean_daily = returns.mean()
    annualized_return = (1 + mean_daily) ** 365 - 1
    sharpe_like = (mean_daily / returns.std() * math.sqrt(365)) if returns.std() > 0 else 0.0
    win_rate = float((returns > 0).mean()) if len(returns) else 0.0
    return {
        "days": len(equity),
        "cumulative_return": cumulative,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "annualized_vol": vol,
        "sharpe_like": sharpe_like,
        "win_rate": win_rate,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top", type=int, default=10, help="number of pools to hold each day (default 10)")
    p.add_argument("--days", type=int, default=30, help="number of most recent snapshots to backtest")
    p.add_argument("--il", choices=["none", "il7d", "heuristic"], default="none", help="apply IL penalty mode")
    p.add_argument("--min-tvl", type=float, default=0, help="minimum TVL filter (USD)")
    p.add_argument("--csv", type=Path, help="optional path to write equity curve CSV")
    p.add_argument("--stable", action="store_true", help="only include stable-stable pools")
    p.add_argument("--lst", action="store_true", help="only include LST/ETH style pools")
    p.add_argument("--wrapper", action="store_true", help="only include wrapper pairs (wBTC/tBTC, wETH/ETH, etc.)")
    p.add_argument("--index", action="store_true", help="only include index/basket style pools")
    p.add_argument("--model", choices=["heuristic", "logit", "xgb"], default="heuristic", help="scoring model for ranking")
    p.add_argument("--retrain", action="store_true", help="retrain the saved logit model before running")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tags = []
    if args.stable:
        tags.append("tag_stable_pegged")
    if args.lst:
        tags.append("tag_lst_pair")
    if args.wrapper:
        tags.append("tag_wrapper_pair")
    if args.index:
        tags.append("tag_index_basket")
    try:
        eq = run_backtest(
            args.top,
            args.days,
            args.il,
            args.min_tvl,
            args.csv,
            include_tags=tags or None,
            model_type=args.model,
            retrain=args.retrain,
        )
    except (RuntimeError, XGBUnavailableError, XGBTrainingError) as exc:
        print(f"Error: {exc}")
        return 1
    summary = summarize(eq)
    if summary:
        print(f"Summary ({DEFAULT_CHAIN} only):", summary)
    else:
        print("No data to backtest.")
    if not eq.empty:
        print(eq.tail())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
