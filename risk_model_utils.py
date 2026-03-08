"""Shared helpers for pure-risk model training."""
from __future__ import annotations

import pandas as pd

from ai_yield_tools import filter_chain, prepare_features
from cache_utils import list_snapshots, load_snapshot

FULL_FEATURES = [
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

# Chosen to avoid the strongest collinearity clusters while preserving
# liquidity, volatility, age, missing-data quality, and IL-style risk.
LOW_COLLINEARITY_FEATURES = [
    "liquidity_depth",
    "token_volatility",
    "age_of_pool",
    "volume_missing_penalty",
    "il_penalty",
]


def join_key(df: pd.DataFrame) -> pd.Series:
    if "pool" in df.columns:
        return df["pool"].astype(str)
    return (
        df.get("project", "").astype(str)
        + "|"
        + df.get("chain", "").astype(str)
        + "|"
        + df.get("symbol", "").astype(str)
    )


def build_risk_event_label(merged: pd.DataFrame) -> pd.Series:
    """Label next-day adverse outcomes as binary risk events."""
    next_missing = merged["_next_missing"].fillna(True)
    curr_tvl = pd.to_numeric(merged.get("tvlUsd"), errors="coerce")
    next_tvl = pd.to_numeric(merged.get("tvlUsd_next"), errors="coerce")
    curr_apy = pd.to_numeric(merged.get("apy"), errors="coerce")
    next_apy = pd.to_numeric(merged.get("apy_next"), errors="coerce")
    curr_il = pd.to_numeric(merged.get("il_penalty"), errors="coerce").fillna(0)
    next_il = pd.to_numeric(merged.get("il_penalty_next"), errors="coerce").fillna(curr_il)
    next_vol = pd.to_numeric(merged.get("token_volatility_next"), errors="coerce").fillna(0)

    tvl_drop = (next_tvl - curr_tvl) / curr_tvl.replace(0, pd.NA)
    apy_drop = curr_apy - next_apy
    il_jump = next_il - curr_il

    risk_event = (
        next_missing
        | (tvl_drop <= -0.30).fillna(False)
        | (apy_drop >= 15).fillna(False)
        | (next_il >= 1.75).fillna(False)
        | (il_jump >= 0.75).fillna(False)
        | (next_vol >= 1.25).fillna(False)
    )
    return risk_event.astype(int)


def build_training_frame_from_cache(
    *,
    feature_names: list[str],
    max_pairs: int = 90,
) -> tuple[pd.DataFrame, pd.Series]:
    snaps = list_snapshots()
    if len(snaps) < 2:
        raise RuntimeError("Need at least 2 cached snapshots to train a risk model.")

    pairs = list(zip(snaps[:-1], snaps[1:]))[-max_pairs:]
    x_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []

    for p_t, p_next in pairs:
        df_t = prepare_features(filter_chain(load_snapshot(p_t)))
        df_next = prepare_features(filter_chain(load_snapshot(p_next)))
        if df_t.empty or df_next.empty:
            continue

        left = df_t.copy()
        right = df_next.copy()
        left["_join_key"] = join_key(left)
        right["_join_key"] = join_key(right)
        next_cols = [
            "_join_key",
            "apy",
            "tvlUsd",
            "il_penalty",
            "token_volatility",
        ]
        merged = left.merge(
            right[next_cols].rename(columns={col: f"{col}_next" for col in next_cols if col != "_join_key"}),
            on="_join_key",
            how="left",
        )
        if merged.empty:
            continue

        merged["_next_missing"] = merged["apy_next"].isna() & merged["tvlUsd_next"].isna()
        x = merged[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = build_risk_event_label(merged)
        x_parts.append(x)
        y_parts.append(y)

    if not x_parts:
        raise RuntimeError("No aligned pool history across snapshots for supervised training.")

    X = pd.concat(x_parts, axis=0, ignore_index=True)
    y = pd.concat(y_parts, axis=0, ignore_index=True)
    if len(X) < 200:
        raise RuntimeError(f"Not enough training rows ({len(X)}). Need at least 200.")
    if y.nunique() < 2:
        raise RuntimeError("Risk labels collapsed to a single class; need more varied cache history.")
    return X, y


__all__ = [
    "FULL_FEATURES",
    "LOW_COLLINEARITY_FEATURES",
    "build_risk_event_label",
    "build_training_frame_from_cache",
    "join_key",
]
