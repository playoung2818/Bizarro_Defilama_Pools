"""XGBoost-based scoring pipeline for DeFi pools."""
from __future__ import annotations

import math
import typing as t

import pandas as pd

from ai_yield_tools import prepare_features
from cache_utils import list_snapshots, load_snapshot

FEATURES = [
    "apy",
    "tvlUsd",
    "volumeUsd7d",
    "volumeUsd1d",
    "liquidity_depth",
    "token_volatility",
    "volume_missing_penalty",
    "age_of_pool",
    "smart_contract_risk",
    "il_penalty",
]


class XGBUnavailableError(RuntimeError):
    """Raised when xgboost dependency is missing."""


class XGBTrainingError(RuntimeError):
    """Raised when cache data is insufficient for training."""


def _daily_rate_from_apy(apy: float) -> float:
    if apy is None or math.isnan(apy):
        return 0.0
    return (1 + apy / 100) ** (1 / 365) - 1


def _build_label(df_next: pd.DataFrame) -> pd.Series:
    """Use next-day realized APY net of IL penalty proxy as supervised label."""
    apy = pd.to_numeric(df_next.get("apy"), errors="coerce")
    il_penalty = pd.to_numeric(df_next.get("il_penalty"), errors="coerce").fillna(0)
    daily = apy.map(_daily_rate_from_apy)
    # Keep IL effect small on daily label scale.
    il_drag = il_penalty * 0.0002
    return daily - il_drag


def _join_key(df: pd.DataFrame) -> pd.Series:
    if "pool" in df.columns:
        return df["pool"].astype(str)
    return (
        df.get("project", "").astype(str)
        + "|"
        + df.get("chain", "").astype(str)
        + "|"
        + df.get("symbol", "").astype(str)
    )


def build_training_frame_from_cache(max_pairs: int = 90) -> tuple[pd.DataFrame, pd.Series]:
    snaps = list_snapshots()
    if len(snaps) < 2:
        raise XGBTrainingError("Need at least 2 cached snapshots to train XGBoost model.")

    pairs = list(zip(snaps[:-1], snaps[1:]))[-max_pairs:]
    x_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []

    for p_t, p_next in pairs:
        df_t = prepare_features(load_snapshot(p_t))
        df_next = prepare_features(load_snapshot(p_next))
        if df_t.empty or df_next.empty:
            continue

        left = df_t.copy()
        right = df_next.copy()
        left["_join_key"] = _join_key(left)
        right["_join_key"] = _join_key(right)
        right["_target"] = _build_label(right)

        merged = left.merge(right[["_join_key", "_target"]], on="_join_key", how="inner")
        if merged.empty:
            continue
        x = merged[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = pd.to_numeric(merged["_target"], errors="coerce").fillna(0)
        x_parts.append(x)
        y_parts.append(y)

    if not x_parts:
        raise XGBTrainingError("No aligned pool history across snapshots for supervised training.")
    X = pd.concat(x_parts, axis=0, ignore_index=True)
    y = pd.concat(y_parts, axis=0, ignore_index=True)
    if len(X) < 200:
        raise XGBTrainingError(f"Not enough training rows ({len(X)}). Need at least 200.")
    return X, y


def train_xgb_from_cache(max_pairs: int = 90):
    try:
        from xgboost import XGBRegressor
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

    X, y = build_training_frame_from_cache(max_pairs=max_pairs)
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )
    model.fit(X, y)
    return model


def score_pools_xgb(df: pd.DataFrame, model) -> pd.DataFrame:
    """Predict final_score with trained XGBoost model."""
    work = prepare_features(df)
    if work.empty:
        return work
    X = work[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    preds = model.predict(X)
    work["final_score"] = preds
    # keep risk_score column for display parity; xgb score is the primary sort key
    work["risk_score"] = work["final_score"]
    return work.sort_values(["final_score", "risk_score", "apy"], ascending=[False, False, False])
