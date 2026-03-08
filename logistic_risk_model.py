#!/usr/bin/env python3
"""Train and inspect a low-collinearity logistic risk model for Ethereum pools."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_yield_tools import prepare_features, top_safe_apys
from cache_utils import ensure_today_snapshot, load_snapshot
from risk_model_utils import LOW_COLLINEARITY_FEATURES, build_training_frame_from_cache

DEFAULT_MODEL_PATH = Path("data/logit_risk_model.pkl")


def train_logistic_risk_model(max_pairs: int = 90) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    X, y = build_training_frame_from_cache(
        feature_names=LOW_COLLINEARITY_FEATURES,
        max_pairs=max_pairs,
    )
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model, X, y


def save_logistic_risk_model(model: Pipeline, path: Path = DEFAULT_MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(model, fh)
    return path


def load_logistic_risk_model(path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    with path.open("rb") as fh:
        model = pickle.load(fh)
    if not isinstance(model, Pipeline):
        raise RuntimeError(f"Saved model at {path} is not a sklearn Pipeline.")
    return model


def get_logistic_risk_model(
    *,
    retrain: bool = False,
    max_pairs: int = 90,
    path: Path = DEFAULT_MODEL_PATH,
) -> tuple[Pipeline, bool]:
    if not retrain and path.exists():
        return load_logistic_risk_model(path), False
    model, _, _ = train_logistic_risk_model(max_pairs=max_pairs)
    save_logistic_risk_model(model, path)
    return model, True


def score_pools_logistic(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    work = prepare_features(df)
    if work.empty:
        return work
    X = work[LOW_COLLINEARITY_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    risk_prob = pd.Series(model.predict_proba(X)[:, 1], index=work.index)
    work["risk_probability"] = risk_prob
    work["risk_score"] = 1.0 - risk_prob
    work["final_score"] = work["risk_score"]
    return work.sort_values(["risk_score", "apy", "tvlUsd"], ascending=[False, False, False])


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    clf = model.named_steps["logit"]
    rows = []
    for feature, coef in zip(LOW_COLLINEARITY_FEATURES, clf.coef_[0]):
        rows.append(
            {
                "feature": feature,
                "coefficient_zscore": coef,
                "odds_ratio_per_1sd": float(np.exp(coef)),
            }
        )
    return pd.DataFrame(rows).sort_values("coefficient_zscore", ascending=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pairs", type=int, default=90, help="number of snapshot pairs to train on")
    parser.add_argument("--top", type=int, default=10, help="show top N safest pools after scoring")
    parser.add_argument("--retrain", action="store_true", help="retrain and overwrite the saved logistic model")
    args = parser.parse_args(argv)

    model, X, y = train_logistic_risk_model(max_pairs=args.max_pairs)
    if args.retrain or not DEFAULT_MODEL_PATH.exists():
        save_logistic_risk_model(model, DEFAULT_MODEL_PATH)
    scored = score_pools_logistic(load_snapshot(ensure_today_snapshot()), model)
    coefs = coefficient_table(model)

    print("Training rows:", len(X))
    print("Risk-event rate:", round(float(y.mean()), 4))
    print("")
    print("Selected low-collinearity features:")
    print(", ".join(LOW_COLLINEARITY_FEATURES))
    print(f"Saved model path: {DEFAULT_MODEL_PATH}")
    print("")
    print("Coefficient table (standardized features):")
    print(coefs.to_string(index=False))
    print("")
    print(f"Top {args.top} safest pools by logistic risk score:")
    top_view = scored.head(args.top)[["project", "chain", "symbol", "apy", "risk_probability", "risk_score"]]
    print(top_view.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
