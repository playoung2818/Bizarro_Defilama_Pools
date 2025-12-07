# Bizarro DefiLlama Pools Toolkit

This project compares the profitability and stability of liquidity pools on [DeFiLlama](https://defillama.com/yields) and adds tooling to rank safer APYs plus simulate impermanent loss.

### Snapshot Findings (from prior exploration)
- Average APY across pools is ~2.5%, with occasional spikes up to ~50% (e.g., USDT on Aave) before reverting.
- Some popular pools like the [USDC pool on Aave3](https://defillama.com/yields/pool/aa70268e-4b52-42bf-a116-608b370f9501) have been trending more profitable.
- Reference visuals are stored in the earlier notebook outputs (images not re-hosted here).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install pandas numpy matplotlib openai  # openai optional
```

## CLI usage
```bash
python3 ai_yield_cli.py top --top 5          # print Top 5 safest APYs today
python3 ai_yield_cli.py il --csv il_curve.csv  # write IL curve CSV (omit --csv to print head)
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
`final_score = 0.7 * risk_score + 0.3 * apy`  (sorted by risk_score desc, then apy desc)

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
- Snapshots are stored in `data/pools_YYYY-MM-DD.json`. The backtest uses the most recent N snapshots available. If none exist, it fetches today’s first.
