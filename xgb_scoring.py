"""XGBoost-based pure-risk scoring pipeline for DeFi pools."""
from __future__ import annotations

import typing as t

import pandas as pd

from ai_yield_tools import prepare_features
from risk_model_utils import FULL_FEATURES, build_training_frame_from_cache


class XGBUnavailableError(RuntimeError):
    """Raised when xgboost dependency is missing."""


class XGBTrainingError(RuntimeError):
    """Raised when cache data is insufficient for training."""


def train_xgb_from_cache(max_pairs: int = 90):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover - dependency gate
        detail = str(exc)
        hint = "xgboost is not installed. Install with: pip install xgboost"
        if "libomp" in detail or "OpenMP" in detail:
            hint = (
                "xgboost import failed due to missing OpenMP runtime on macOS. "
                "Install with: brew install libomp"
            )
        raise XGBUnavailableError(
            f"{hint}. Original error: {detail}"
        ) from exc

    try:
        X, y = build_training_frame_from_cache(feature_names=FULL_FEATURES, max_pairs=max_pairs)
    except RuntimeError as exc:
        raise XGBTrainingError(str(exc)) from exc
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )
    model.fit(X, y)
    return model


def score_pools_xgb(df: pd.DataFrame, model) -> pd.DataFrame:
    """Predict pure risk scores with trained XGBoost model."""
    work = prepare_features(df)
    if work.empty:
        return work
    X = work[FULL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    risk_prob = pd.Series(model.predict_proba(X)[:, 1], index=work.index)
    work["risk_probability"] = risk_prob
    work["risk_score"] = 1.0 - risk_prob
    work["final_score"] = work["risk_score"]
    return work.sort_values(["risk_score", "apy", "tvlUsd"], ascending=[False, False, False])
