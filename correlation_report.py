#!/usr/bin/env python3
"""Print a feature correlation matrix for the latest cached Ethereum snapshot."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ai_yield_tools import filter_chain, prepare_features
from cache_utils import list_snapshots, load_snapshot
from risk_model_utils import FULL_FEATURES, LOW_COLLINEARITY_FEATURES


def build_correlation_matrix(snapshot: Path | None = None) -> pd.DataFrame:
    snaps = list_snapshots()
    if snapshot is None:
        if not snaps:
            raise RuntimeError("No cached snapshots available.")
        snapshot = snaps[-1]
    df = filter_chain(load_snapshot(snapshot))
    features = prepare_features(df)
    return features[FULL_FEATURES].apply(pd.to_numeric, errors="coerce").corr()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="optional path to write the correlation matrix as CSV")
    args = parser.parse_args(argv)

    corr = build_correlation_matrix()
    print("Full feature correlation matrix:")
    print(corr.round(3).to_string())
    print("")
    print("Recommended low-collinearity feature subset:")
    print(", ".join(LOW_COLLINEARITY_FEATURES))

    if args.csv:
        corr.to_csv(args.csv)
        print("")
        print(f"Wrote correlation matrix to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
