#!/usr/bin/env python3
"""Command-line helper for AI yield ranking and impermanent loss curves."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from ai_yield_tools import FetchError, fetch_pools_df, score_pools, top_safe_apys
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p_top = sub.add_parser("top", help="Show Top N safe APYs")
    p_top.add_argument("--top", type=int, default=5, help="Number of pools to show (default: 5)")
    p_top.set_defaults(func=cmd_top)

    p_il = sub.add_parser("il", help="Output impermanent loss curve")
    p_il.add_argument("--csv", help="Optional path to write IL curve CSV")
    p_il.set_defaults(func=cmd_il)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
