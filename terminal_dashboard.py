"""Terminal dashboard for DeFiLlama risk/yield monitoring."""
from __future__ import annotations

from datetime import timezone
import time

import pandas as pd

from ai_yield_tools import score_pools, tag_pools, top_safe_apys
from backtest import run_backtest, summarize
from cache_utils import ensure_today_snapshot, load_snapshot
from xgb_scoring import XGBTrainingError, XGBUnavailableError, score_pools_xgb, train_xgb_from_cache

TAG_MAP = {
    "stable": "tag_stable_pegged",
    "lst": "tag_lst_pair",
    "wrapper": "tag_wrapper_pair",
    "index": "tag_index_basket",
}


def _fmt_pct(x: float) -> str:
    return f"{x * 100:,.2f}%"


def _fmt_num(x: float) -> str:
    return f"{x:,.2f}"


def _fmt_usd(x: float) -> str:
    return f"${x:,.0f}"


def _select_tags(tag_filters: list[str] | None) -> list[str]:
    if not tag_filters:
        return []
    return [TAG_MAP[k] for k in tag_filters if k in TAG_MAP]


def build_dashboard(
    *,
    top_n: int = 10,
    lookback_days: int = 30,
    min_tvl: float = 1_000_000,
    il_mode: str = "heuristic",
    model_type: str = "heuristic",
    tag_filters: list[str] | None = None,
) -> str:
    snap_path = ensure_today_snapshot()
    current_df = load_snapshot(snap_path)
    if model_type == "xgb":
        model = train_xgb_from_cache(max_pairs=max(lookback_days, 30))
        scored = score_pools_xgb(current_df, model)
    else:
        scored = score_pools(current_df)
    tagged = tag_pools(scored)

    selected_tags = _select_tags(tag_filters)
    if selected_tags:
        mask = tagged[selected_tags].any(axis=1)
        universe = tagged.loc[mask].copy()
        universe_label = f"Tagged only ({', '.join(tag_filters or [])})"
    else:
        universe = tagged.copy()
        universe_label = "All pools"

    if min_tvl > 0:
        universe = universe[universe["tvlUsd"] >= min_tvl]

    top_df = top_safe_apys(universe, n=top_n).copy()
    eq = run_backtest(
        top_n=top_n,
        days=lookback_days,
        il_mode=il_mode,
        min_tvl=min_tvl,
        csv_path=None,
        include_tags=selected_tags or None,
        model_type=model_type,
    )
    stats = summarize(eq)

    tag_counts = {
        "stable": int(tagged["tag_stable_pegged"].sum()) if "tag_stable_pegged" in tagged else 0,
        "lst": int(tagged["tag_lst_pair"].sum()) if "tag_lst_pair" in tagged else 0,
        "wrapper": int(tagged["tag_wrapper_pair"].sum()) if "tag_wrapper_pair" in tagged else 0,
        "index": int(tagged["tag_index_basket"].sum()) if "tag_index_basket" in tagged else 0,
    }

    lines: list[str] = []
    now = pd.Timestamp.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append("=" * 96)
    lines.append(f"DeFi Risk Dashboard | {now}")
    lines.append(
        f"Snapshot: {snap_path.name} | Universe: {universe_label} | Min TVL: {_fmt_usd(min_tvl)} | "
        f"IL mode: {il_mode} | Model: {model_type}"
    )
    lines.append("=" * 96)
    lines.append(
        f"Pool coverage: {len(tagged):,} total | {len(universe):,} after filters | tags -> "
        f"stable {tag_counts['stable']:,}, lst {tag_counts['lst']:,}, wrapper {tag_counts['wrapper']:,}, index {tag_counts['index']:,}"
    )
    lines.append("Ranking rule: final_score = 0.5 * risk_score + 0.5 * apy, with APY < 4% floored to 0 before scoring.")

    if stats:
        lines.append(
            "Backtest (Top-N rotation): "
            f"days {int(stats['days'])}, cum {_fmt_pct(stats['cumulative_return'])}, "
            f"ann {_fmt_pct(stats['annualized_return'])}, vol {_fmt_pct(stats['annualized_vol'])}, "
            f"mdd {_fmt_pct(stats['max_drawdown'])}, sharpe-like {_fmt_num(stats['sharpe_like'])}, "
            f"win-rate {_fmt_pct(stats['win_rate'])}"
        )
    else:
        lines.append("Backtest: no equity history available.")

    lines.append("-" * 96)
    lines.append(f"Top {min(top_n, len(top_df))} candidates (sorted by final_score desc)")
    if top_df.empty:
        lines.append("No pools found with current filters.")
    else:
        display = top_df.copy()
        if "apy" in display:
            display["apy"] = display["apy"].map(lambda x: f"{x:,.2f}%")
        if "tvlUsd" in display:
            display["tvlUsd"] = display["tvlUsd"].map(lambda x: _fmt_usd(float(x)) if pd.notna(x) else "")
        if "risk_score" in display:
            display["risk_score"] = display["risk_score"].map(lambda x: f"{x:,.2f}")
        if "final_score" in display:
            display["final_score"] = display["final_score"].map(lambda x: f"{x:,.2f}")
        lines.append(display.to_string(index=False))

    if not eq.empty:
        lines.append("-" * 96)
        lines.append("Recent equity tail")
        tail = eq.tail(5).copy()
        if "value" in tail:
            tail["value"] = tail["value"].map(lambda x: f"{x:,.4f}")
        if "daily_ret" in tail:
            tail["daily_ret"] = tail["daily_ret"].map(lambda x: f"{x * 100:,.3f}%")
        lines.append(tail.to_string(index=False))

    return "\n".join(lines)


def run_dashboard_loop(
    *,
    top_n: int,
    lookback_days: int,
    min_tvl: float,
    il_mode: str,
    model_type: str,
    tag_filters: list[str] | None,
    refresh_seconds: int,
    once: bool,
) -> int:
    while True:
        print("\033[2J\033[H", end="")  # clear terminal
        try:
            output = build_dashboard(
                top_n=top_n,
                lookback_days=lookback_days,
                min_tvl=min_tvl,
                il_mode=il_mode,
                model_type=model_type,
                tag_filters=tag_filters,
            )
        except (XGBUnavailableError, XGBTrainingError) as exc:
            output = f"Error: {exc}"
            print(output)
            return 1
        print(output)
        if once:
            return 0
        time.sleep(max(3, refresh_seconds))
