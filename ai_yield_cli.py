#!/usr/bin/env python3
"""Command-line helper for AI yield ranking and impermanent loss curves."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from ai_yield_tools import (
    FetchError,
    fetch_pools_df,
    score_pools,
    tag_pools,
    top_safe_apys,
)
from il_tools import il_curve
from terminal_dashboard import run_dashboard_loop
from xgb_scoring import XGBTrainingError, XGBUnavailableError, score_pools_xgb, train_xgb_from_cache


def _rank_with_model(args: argparse.Namespace, pools_df: pd.DataFrame) -> pd.DataFrame:
    if args.model == "xgb":
        model = train_xgb_from_cache(max_pairs=90)
        return score_pools_xgb(pools_df, model)
    return score_pools(pools_df)


def _fmt_tokens(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if pd.isna(value):
        return ""
    return str(value)


def _format_compact_top(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()
    if "pool" in view.columns:
        view["pool"] = view["pool"].astype(str).str.slice(0, 8)
    if "apy" in view.columns:
        view["apy"] = pd.to_numeric(view["apy"], errors="coerce").round(2)
    if "tvlUsd" in view.columns:
        view["tvlUsd"] = pd.to_numeric(view["tvlUsd"], errors="coerce").map(
            lambda x: f"{x:,.0f}" if pd.notna(x) else ""
        )
    if "risk_score" in view.columns:
        view["risk_score"] = pd.to_numeric(view["risk_score"], errors="coerce").round(3)
    if "final_score" in view.columns:
        view["final_score"] = pd.to_numeric(view["final_score"], errors="coerce").round(3)
    if "url" in view.columns:
        view["url"] = view["url"].astype(str).str.slice(0, 60)

    cols = [
        "pool",
        "project",
        "chain",
        "symbol",
        "apy",
        "tvlUsd",
        "risk_score",
        "final_score",
        "url",
    ]
    cols = [c for c in cols if c in view.columns]
    return view[cols]


def _print_stake_guide(row: pd.Series) -> None:
    print("=" * 90)
    print(f"Pool index: {row.name}")
    print(f"Pool id: {row.get('pool', '')}")
    print(f"Project/Chain: {row.get('project', '')} / {row.get('chain', '')}")
    print(f"Symbol: {row.get('symbol', '')}")
    print(f"APY: {row.get('apy', '')}")
    print(f"TVL(USD): {row.get('tvlUsd', '')}")
    print(f"URL: {row.get('url', '')}")
    print(f"Underlying tokens: {_fmt_tokens(row.get('underlyingTokens'))}")
    pool_meta = row.get("poolMeta", "")
    if isinstance(pool_meta, str) and pool_meta.strip():
        print(f"Pool meta: {pool_meta}")
    print("")
    print("How to stake:")
    print("1. Open the URL above (or the protocol app page for this pool).")
    print("2. Connect wallet on the same chain shown above.")
    print("3. If this is a vault over an LP token, first mint/get the LP token from the source AMM.")
    print("4. Approve token spending, then Deposit/Stake.")
    print("5. Verify position in your wallet/protocol dashboard and monitor exit liquidity.")


def cmd_top(args: argparse.Namespace) -> int:
    try:
        pools_df = fetch_pools_df()
    except FetchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        ranked = _rank_with_model(args, pools_df)
    except (XGBUnavailableError, XGBTrainingError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    topn = top_safe_apys(ranked, n=args.top)
    pd.set_option("display.max_rows", args.top)
    if args.full:
        print(topn.to_string())
    else:
        print(_format_compact_top(topn).to_string())
    return 0


def cmd_il(args: argparse.Namespace) -> int:
    curve = il_curve()
    path = args.csv
    if path:
        curve.to_csv(path, index=False)
        print(f"Wrote IL curve to {path}")
    else:
        print(curve.head())
    return 0


def cmd_find(args: argparse.Namespace) -> int:
    try:
        pools_df = fetch_pools_df()
    except FetchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    tagged = tag_pools(pools_df)
    mask = False
    tags = []
    if args.stable:
        tags.append("tag_stable_pegged")
    if args.lst:
        tags.append("tag_lst_pair")
    if args.wrapper:
        tags.append("tag_wrapper_pair")
    if args.index:
        tags.append("tag_index_basket")
    if not tags:
        tags = ["tag_stable_pegged", "tag_lst_pair", "tag_wrapper_pair", "tag_index_basket"]
    mask = tagged[tags].any(axis=1)
    result = tagged.loc[mask, ["project", "chain", "symbol"] + tags][: args.limit]
    print(result)
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    tag_filters = []
    if args.stable:
        tag_filters.append("stable")
    if args.lst:
        tag_filters.append("lst")
    if args.wrapper:
        tag_filters.append("wrapper")
    if args.index:
        tag_filters.append("index")
    return run_dashboard_loop(
        top_n=args.top,
        lookback_days=args.days,
        min_tvl=args.min_tvl,
        il_mode=args.il,
        model_type=args.model,
        tag_filters=tag_filters or None,
        refresh_seconds=args.refresh,
        once=args.once,
    )


def cmd_stake(args: argparse.Namespace) -> int:
    try:
        pools_df = fetch_pools_df()
    except FetchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        ranked = _rank_with_model(args, pools_df)
    except (XGBUnavailableError, XGBTrainingError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    row = None
    if args.index is not None:
        try:
            row = ranked.loc[int(args.index)]
        except KeyError:
            print(f"Error: index {args.index} not found in current ranked DataFrame.", file=sys.stderr)
            return 1
    elif args.pool is not None:
        if "pool" not in ranked.columns:
            print("Error: pool id column is missing in fetched data.", file=sys.stderr)
            return 1
        matches = ranked[ranked["pool"].astype(str) == str(args.pool)]
        if matches.empty:
            print(f"Error: pool id {args.pool} not found.", file=sys.stderr)
            return 1
        row = matches.iloc[0]
    else:
        row = ranked.iloc[0]

    _print_stake_guide(row)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p_top = sub.add_parser("top", help="Show Top N safe APYs")
    p_top.add_argument("--top", type=int, default=5, help="Number of pools to show (default: 5)")
    p_top.add_argument("--model", choices=["heuristic", "xgb"], default="heuristic", help="scoring model")
    p_top.add_argument("--full", action="store_true", help="show full raw columns")
    p_top.set_defaults(func=cmd_top)

    p_il = sub.add_parser("il", help="Output impermanent loss curve")
    p_il.add_argument("--csv", help="Optional path to write IL curve CSV")
    p_il.set_defaults(func=cmd_il)

    p_find = sub.add_parser("find", help="Find LPs matching pegged/wrapper/index traits")
    p_find.add_argument("--stable", action="store_true", help="stable-stable pairs")
    p_find.add_argument("--lst", action="store_true", help="LST vs ETH pairs")
    p_find.add_argument("--wrapper", action="store_true", help="wrapper of same base asset (e.g., WBTC/tBTC)")
    p_find.add_argument("--index", action="store_true", help="index/basket style pools")
    p_find.add_argument("--limit", type=int, default=20, help="rows to display")
    p_find.set_defaults(func=cmd_find)

    p_dash = sub.add_parser("dashboard", help="Run terminal risk dashboard")
    p_dash.add_argument("--top", type=int, default=10, help="Top N pools to show")
    p_dash.add_argument("--days", type=int, default=30, help="Lookback days for backtest summary")
    p_dash.add_argument("--min-tvl", type=float, default=1_000_000, help="Minimum TVL filter (USD)")
    p_dash.add_argument("--il", choices=["none", "il7d", "heuristic"], default="heuristic", help="IL penalty mode")
    p_dash.add_argument("--model", choices=["heuristic", "xgb"], default="heuristic", help="scoring model")
    p_dash.add_argument("--stable", action="store_true", help="filter to stable-stable pools")
    p_dash.add_argument("--lst", action="store_true", help="filter to LST/ETH pools")
    p_dash.add_argument("--wrapper", action="store_true", help="filter to wrapper pairs")
    p_dash.add_argument("--index", action="store_true", help="filter to index/basket pools")
    p_dash.add_argument("--refresh", type=int, default=30, help="refresh interval in seconds")
    p_dash.add_argument("--once", action="store_true", help="print once and exit")
    p_dash.set_defaults(func=cmd_dashboard)

    p_stake = sub.add_parser("stake", help="Show staking guide for a selected pool")
    p_stake.add_argument("--model", choices=["heuristic", "xgb"], default="heuristic", help="scoring model")
    p_stake.add_argument("--index", type=int, help="DataFrame index from `top` output (example: 14811)")
    p_stake.add_argument("--pool", help="DeFiLlama pool id")
    p_stake.set_defaults(func=cmd_stake)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
