#!/usr/bin/env python3
"""Persist logistic risk-model training metrics and coefficients for drift tracking."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from logistic_risk_model import DEFAULT_MODEL_PATH, coefficient_table, save_logistic_risk_model, train_logistic_risk_model
from risk_model_utils import LOW_COLLINEARITY_FEATURES

DEFAULT_OUTPUT = Path("data/model_metrics_logit.csv")


def build_metrics_row(max_pairs: int) -> dict[str, float | int | str]:
    model, X, y = train_logistic_risk_model(max_pairs=max_pairs)
    save_logistic_risk_model(model, DEFAULT_MODEL_PATH)
    coefs = coefficient_table(model).set_index("feature")
    clf = model.named_steps["logit"]

    row: dict[str, float | int | str] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "logit",
        "max_pairs": max_pairs,
        "training_rows": int(len(X)),
        "event_rate": float(y.mean()),
        "positive_count": int(y.sum()),
        "negative_count": int((1 - y).sum()),
        "intercept_zscore": float(clf.intercept_[0]),
        "features": ",".join(LOW_COLLINEARITY_FEATURES),
    }

    for feature in LOW_COLLINEARITY_FEATURES:
        row[f"coef_{feature}"] = float(coefs.loc[feature, "coefficient_zscore"])
        row[f"odds_ratio_{feature}"] = float(coefs.loc[feature, "odds_ratio_per_1sd"])
    return row


def append_metrics_row(path: Path, row: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if path.exists():
        history = pd.read_csv(path)
        frame = pd.concat([history, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pairs", type=int, default=90, help="number of snapshot pairs to train on")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="CSV path for saved training metrics")
    args = parser.parse_args(argv)

    row = build_metrics_row(args.max_pairs)
    append_metrics_row(args.out, row)

    print(f"Saved metrics to {args.out}")
    print(f"timestamp_utc: {row['timestamp_utc']}")
    print(f"training_rows: {row['training_rows']}")
    print(f"event_rate: {row['event_rate']:.4f}")
    print(f"intercept_zscore: {row['intercept_zscore']:.6f}")
    print(f"saved_model: {DEFAULT_MODEL_PATH}")
    print("coefficients:")
    for feature in LOW_COLLINEARITY_FEATURES:
        print(f"  {feature}: {row[f'coef_{feature}']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
