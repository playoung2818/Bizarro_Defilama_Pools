# Bizarro DefiLlama Pools Toolkit

This project compares the profitability and stability of [Ethereum](https://ethereum.org/) liquidity pools on [DeFiLlama](https://defillama.com/yields) and adds tooling to rank safer APYs plus simulate impermanent loss.

### Snapshot Findings (from prior exploration)
- Average APY across pools is ~2.5%, with occasional spikes up to ~50% (e.g., USDT on Aave) before reverting.
- Some popular pools like the [USDC pool on Aave3](https://defillama.com/yields/pool/aa70268e-4b52-42bf-a116-608b370f9501) have been trending more profitable.
- Reference visuals are stored in the earlier notebook outputs (images not re-hosted here).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install pandas numpy matplotlib openai xgboost  # openai/xgboost optional
```

## CLI usage
```bash
python3 ai_yield_cli.py top --top 5          # print Top 5 safest Ethereum APYs today
python3 ai_yield_cli.py top --top 10 --model logit # retrain logistic coefficients and rank Ethereum pools
python3 ai_yield_cli.py top --top 10 --model logit --retrain # same effect; explicit retrain flag
python3 ai_yield_cli.py top --top 10 --model xgb   # XGBoost pure-risk model on Ethereum pools (needs cached history)
python3 ai_yield_cli.py il --csv il_curve.csv  # write IL curve CSV (omit --csv to print head)
python3 ai_yield_cli.py find --limit 20        # show LPs tagged as pegged/wrapper/index; add --stable/--lst/--wrapper/--index to focus
python3 ai_yield_cli.py dashboard --once       # one-shot terminal dashboard
python3 ai_yield_cli.py dashboard --once --model logit # one-shot dashboard using saved logistic pure-risk model
python3 ai_yield_cli.py dashboard --once --model xgb  # one-shot dashboard using XGBoost pure-risk model
python3 ai_yield_cli.py dashboard              # live dashboard (refresh every 30s)
python3 ai_yield_cli.py dashboard --stable --top 15 --days 60 --min-tvl 5000000 --il heuristic
python3 correlation_report.py                  # print latest Ethereum feature correlation matrix
python3 logistic_risk_model.py --top 10 --retrain # train, save, and inspect the logistic risk model
python3 save_model_metrics.py                  # retrain, save, and append coefficient drift metrics
```

## Notebook usage
Open `defillama_api_yields.ipynb` and run the new sections:
- **AI Yield Aggregator**: pulls pools from `https://yields.llama.fi/pools`, scores them, and shows the Top 5 safe APYs.
- **LLM justification (optional)**: needs `openai` and `OPENAI_API_KEY`; returns short rationale bullets.
- **Impermanent Loss Simulator**: plots IL vs. price change and prints single-point IL values.

## How scoring works (in `ai_yield_tools.py`)
- `liquidity_depth = log10(tvlUsd + 1)` (higher TVL → safer)
- `token_volatility = volumeProxy / (tvlUsd + 1)` where `volumeProxy = volumeUsd7d` or `7 * volumeUsd1d` when 7d is missing; if both volumes are missing we treat volatility as high (5) and add a penalty (lower → safer, clipped at 5)
- `age_of_pool` = days since `apyBaseInception` (older → safer)
- `smart_contract_risk` from `ilRisk` (`yes`=1, `no`=0, unknown=0.5)
- `il_penalty` (impermanent-loss risk) from `stablecoin`, `exposure`, `il7d`, and volatility:
  - stablecoin: 0
  - exposure single: 0; hedged: 0.3; multi: 1.0; other/unknown: 0.6
  - plus `5 * max(il7d, 0)` and `0.1 * min(token_volatility, 5)` as small amplifiers

`risk_score` (higher = safer):
```
-1.2 * token_volatility
+0.8 * liquidity_depth
+0.5 * ln(1 + age_of_pool_days)
-1.0 * smart_contract_risk
-1.0 * volume_missing_penalty
-1.0 * il_penalty
```
The pool universe is filtered to `3.25 < apy <= 50` before scoring. In `heuristic` mode, `final_score = risk_score`, so ranking is purely risk-driven. Pools are sorted by `risk_score` desc, then `apy` desc, then `tvlUsd` desc as tie-breakers.

## Impermanent Loss functions (in `il_tools.py`)
- `il_curve(price_changes)`: DataFrame of IL% vs price change for a 50/50 AMM.
- `il_at_change(price_change)`: single-point IL% for a fractional move (e.g., `-0.5` for -50%).

## Files
- `ai_yield_tools.py`: fetch/score pools; optional LLM explanations.
- `il_tools.py`: IL curve math.
- `ai_yield_cli.py`: CLI for rankings and IL CSV export.
- `defillama_api_yields.ipynb`: notebook with demos (latest sections appended).
- `defillama_api_yields_historical_data.ipynb`: earlier exploration notebook.
- `cache_utils.py`: cache/fetch daily pool snapshots to `data/`.
- `backtest.py`: simple top-N rotation backtester using cached snapshots.
- `runner.py`: loops daily to cache and backtest (Pi-friendly).
- `correlation_report.py`: prints a correlation matrix for the latest cached Ethereum snapshot.
- `logistic_risk_model.py`: trains a low-collinearity logistic classifier on cached risk events.
- `risk_model_utils.py`: shared risk-label and supervised training-frame builders for logistic/XGBoost models.
- Pool tagging helpers for pegged/wrapper/index live in `ai_yield_tools.tag_pools()` and the CLI `find` command.

## Optional: LLM setup
```bash
export OPENAI_API_KEY=sk-...
```
Then run the optional justification cell in the notebook.

## Backtesting (top-N rotation)
- Cache today’s snapshot: `python3 -m venv .venv && source .venv/bin/activate && python3 -m pip install pandas numpy matplotlib openai` (once), then `python3 -c "import cache_utils; cache_utils.ensure_today_snapshot()"`.
- Run backtest over latest 30 snapshots (or fewer if not present):  
  ```bash
  python3 backtest.py --top 10 --days 30 --il heuristic --min-tvl 1_000_000 --csv equity.csv
  ```
- `--il` modes: `none` (default), `il7d` (use il7d/7 per day when available), `heuristic` (volatility-based haircut).
- `--model` modes: `heuristic` (hand-tuned formula), `logit` (low-collinearity logistic classifier trained from cached risk events), `xgb` (XGBoost classifier trained from cached snapshot history to predict next-day risk events).
- Snapshots are stored in `data/pools_YYYY-MM-DD.json`. The backtest uses the most recent N snapshots available. If none exist, it fetches today’s first.
- Summary metrics include: cumulative return, annualized return, annualized volatility, max drawdown, sharpe-like ratio, and win rate.

## Terminal dashboard
- `dashboard` mode gives a risk-monitoring view directly in terminal:
  - Current Ethereum pool coverage and tag counts (stable/LST/wrapper/index)
  - Backtest risk/return summary (annualized return, vol, MDD, sharpe-like, win rate)
  - Top-N candidate table based on risk-aware ranking
  - Recent equity curve tail
- If live fetch is unavailable, dashboard falls back to the latest cached snapshot in `data/`.

## Pure risk model direction (no return optimization)
If the objective is risk quantification only, use a pure risk pipeline:

- Target variable: future risk event label (for example `risk_event_1d` or `risk_event_7d`), not return.
  - Example: label = 1 if next period has severe drawdown/depeg/liquidity shock, else 0.
- Output: calibrated risk probability `P(risk_event)` and a normalized risk score:
  - `RiskScore = 100 * P(risk_event)`
- Risk tiers:
  - Low: `< 20`
  - Medium: `20 - 50`
  - High: `> 50`
- Validation: out-of-sample walk-forward only (no random split).
- Recommended metrics:
  - `PR-AUC`, `ROC-AUC`
  - `Brier score` (probability quality)
  - calibration curve
  - recall on high-risk bucket

This supports risk-tolerance filtering directly:
- Conservative: `RiskScore <= 20`
- Balanced: `RiskScore <= 35`
- Aggressive: `RiskScore <= 50`

## Daily runner (e.g., on a Pi)
Run continuously to refresh the snapshot and backtest once per day:
```bash
python3 runner.py
```
Customize `top_n`, `days`, `il_mode`, `min_tvl`, or CSV path inside `runner.py` if desired. For cron/systemd, call `cache_utils.ensure_today_snapshot()` then `backtest.py` in a job instead of a long-lived loop.
