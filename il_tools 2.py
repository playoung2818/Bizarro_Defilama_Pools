"""Impermanent loss utilities for 50/50 AMM pairs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def il_curve(price_changes=np.linspace(-0.8, 4.0, 200)) -> pd.DataFrame:
    """
    Return IL (%) over a range of price changes for a 50/50 pool.

    price_changes: array-like of fractional changes (e.g., -0.2 for -20%).
    """
    r = 1 + np.array(price_changes, dtype=float)
    il = 2 * np.sqrt(r) / (1 + r) - 1
    return pd.DataFrame({
        "price_change_pct": r * 100 - 100,
        "il_pct": il * 100,
    })


def il_at_change(price_change: float) -> float:
    """Impermanent loss percent at a single fractional price change."""
    r = 1 + float(price_change)
    if r <= 0:
        return float("nan")
    return float(2 * (r ** 0.5) / (1 + r) - 1) * 100


__all__ = ["il_curve", "il_at_change"]
