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


def cmd_top(args: argparse.Namespace) -> int:
    try:
        pools_df = fetch_pools_df()
    except FetchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    ranked = score_pools(pools_df)
    topn = top_safe_apys(ranked, n=args.top)
    pd.set_option("display.max_rows", args.top)
    print(topn)
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p_top = sub.add_parser("top", help="Show Top N safe APYs")
    p_top.add_argument("--top", type=int, default=5, help="Number of pools to show (default: 5)")
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

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
