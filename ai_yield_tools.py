"""Utilities for fetching and ranking DeFiLlama yield pools."""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import typing as t
import urllib.request

import numpy as np
import pandas as pd

# Endpoint documented at https://yields.llama.fi/pools
YIELDS_URL = "https://yields.llama.fi/pools"


class FetchError(RuntimeError):
    """Raised when a fetch from DeFiLlama fails."""


def fetch_pools_df(timeout: int = 30) -> pd.DataFrame:
    """Return a DataFrame of all pools from DeFiLlama yields API."""
    try:
        with urllib.request.urlopen(YIELDS_URL, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception as exc:  # pragma: no cover - thin wrapper
        raise FetchError(f"Failed to fetch {YIELDS_URL}: {exc}") from exc

    pools = data.get("data", [])
    return pd.DataFrame(pools)


def _safe_log10(series: pd.Series) -> pd.Series:
    return series.fillna(0).astype(float).add(1).apply(math.log10)


def _parse_age(days_series: pd.Series) -> pd.Series:
    # apyBaseInception is ISO timestamp; convert to age in days
    parsed = pd.to_datetime(days_series, errors="coerce", utc=True)
    now = pd.Timestamp(dt.datetime.utcnow(), tz="UTC")
    return (now - parsed).dt.days.fillna(0)


def score_pools(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach heuristic risk and final scores.

    risk_score: higher is safer. Combines liquidity depth, inverse volatility,
    age, and smart-contract risk flag. final_score also rewards APY.
    """
    work = df.copy()

    work["tvlUsd"] = pd.to_numeric(work.get("tvlUsd"), errors="coerce")
    work["volumeUsd7d"] = pd.to_numeric(work.get("volumeUsd7d"), errors="coerce")
    work["apy"] = pd.to_numeric(work.get("apy"), errors="coerce")

    work["liquidity_depth"] = _safe_log10(work["tvlUsd"])
    volume7d = pd.to_numeric(work.get("volumeUsd7d"), errors="coerce")
    volume1d = pd.to_numeric(work.get("volumeUsd1d"), errors="coerce")
    volume_proxy = volume7d.fillna(volume1d * 7)
    vol_missing = volume_proxy.isna()
    work["token_volatility"] = (volume_proxy / (work["tvlUsd"].abs() + 1)).fillna(5)
    work["volume_missing_penalty"] = vol_missing.astype(float) * 1.5
    work["age_of_pool"] = _parse_age(work.get("apyBaseInception"))
    work["smart_contract_risk"] = work.get("ilRisk").map({"yes": 1, "no": 0}).fillna(0.5)

    exposure = work.get("exposure", pd.Series(index=work.index))
    stablecoin = work.get("stablecoin", pd.Series(index=work.index)).fillna(False)
    il7d = pd.to_numeric(work.get("il7d"), errors="coerce").fillna(0)
    il_penalty = []
    for exp, is_stable, il_val, vol in zip(exposure, stablecoin, il7d, work["token_volatility"]):
        if is_stable:
            base = 0.0
        elif isinstance(exp, str) and exp.lower() == "single":
            base = 0.0  # single-sided has negligible IL
        elif isinstance(exp, str) and exp.lower() == "hedged":
            base = 0.3
        elif isinstance(exp, str) and exp.lower() == "multi":
            base = 1.0
        else:
            base = 0.6
        # amplify by recent reported IL and volatility proxy
        penalty = base + 5 * max(il_val, 0) + 0.1 * min(vol, 5)
        il_penalty.append(penalty)
    work["il_penalty"] = pd.Series(il_penalty, index=work.index)

    work["risk_score"] = (
        -1.2 * work["token_volatility"].clip(upper=5)
        + 0.8 * work["liquidity_depth"]
        + 0.5 * np.log1p(work["age_of_pool"])
        - 1.0 * work["smart_contract_risk"]
        - 1.0 * work["volume_missing_penalty"]
        - 1.0 * work["il_penalty"]
    )
    # Require a minimum APY floor and rebalance weights to emphasize yield more.
    apy = work["apy"].fillna(0)
    apy = apy.where(apy >= 4, 0)  # zero out pools with APY < 4%
    work["final_score"] = work["risk_score"] * 0.5 + apy * 0.5
    return work.sort_values(["risk_score", "apy"], ascending=[False, False])


def top_safe_apys(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the top N pools with the highest safety score."""
    cols = [
        "project",
        "chain",
        "symbol",
        "apy",
        "tvlUsd",
        "volumeUsd7d",
        "risk_score",
        "final_score",
    ]
    available_cols = [c for c in cols if c in df.columns]
    return df.head(n)[available_cols]


def llm_rank_explanations(
    ranked_df: pd.DataFrame,
    client: t.Any,
    model: str = "gpt-4o-mini",
    top_n: int = 5,
) -> t.Optional[t.List[str]]:
    """
    Optional helper to ask an LLM for short rationales on the ranking.

    client: an OpenAI-compatible client with .chat.completions.create().
    Returns a list of bullet strings or None if call fails.
    """
    subset = ranked_df.head(top_n)
    records = subset[[c for c in ["project", "chain", "symbol", "apy", "risk_score", "tvlUsd"] if c in subset.columns]]
    prompt = (
        "You are a risk-focused DeFi analyst. Given these pools with APY, safety score, "
        "and TVL, justify why each is in the Top 5 safest APYs today. Keep each bullet short.\n"
        f"Pools:\n{records.to_json(orient='records')}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300,
        )
    except Exception:
        return None

    content = resp.choices[0].message["content"] if hasattr(resp.choices[0], "message") else None
    if not content:
        return None
    lines = [line.strip("- ") for line in content.splitlines() if line.strip()]
    return lines or None


__all__ = [
    "FetchError",
    "fetch_pools_df",
    "score_pools",
    "top_safe_apys",
    "llm_rank_explanations",
]
