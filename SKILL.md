---
name: defillama-risk-ranking
description: Use this skill when working on the Bizarro_Defilama_Pools project to inspect, train, explain, or modify the Ethereum-only risk-ranking workflow, including heuristic, logistic, and XGBoost models.
---

# DeFiLlama Risk Ranking

This skill is for `/Users/a20748/Desktop/Bizarro_Defilama_Pools`.

## Current system

- Universe is filtered to `chain = Ethereum`.
- Pools are filtered to `3.25 < apy <= 50` before scoring.
- Default day-to-day command:
  - `python ai_yield_cli.py top --top 10`
- Heuristic model:
  - `python ai_yield_cli.py top --top 10`
  - Pure risk ranking.
  - `final_score = risk_score`
- Logistic model:
  - `python ai_yield_cli.py top --top 10 --model logit`
  - For `top`, coefficients retrain every run.
  - For `dashboard` and `backtest`, the saved model is used by default unless `--retrain` is passed.
- XGBoost model:
  - `python ai_yield_cli.py top --top 10 --model xgb`
  - Pure-risk classifier trained from cached snapshots.

## Main files

- `ai_yield_cli.py`: top/find/dashboard/stake CLI entrypoint.
- `ai_yield_tools.py`: shared fetch/filter/feature/heuristic ranking logic.
- `risk_model_utils.py`: shared risk labels and training-frame builder.
- `logistic_risk_model.py`: logistic risk model training, saving, loading, scoring.
- `xgb_scoring.py`: XGBoost pure-risk scoring.
- `backtest.py`: historical backtest runner.
- `terminal_dashboard.py`: dashboard output.
- `save_model_metrics.py`: append training metrics and coefficients for drift tracking.
- `correlation_report.py`: feature correlation matrix on cached Ethereum data.

## Commands to remember

- Quick heuristic ranking:
  - `python ai_yield_cli.py top --top 10`
- Retrained logistic ranking:
  - `python ai_yield_cli.py top --top 10 --model logit`
- Logistic dashboard using saved model:
  - `python ai_yield_cli.py dashboard --once --model logit`
- Logistic backtest using saved model:
  - `python backtest.py --model logit --days 30 --top 10`
- Correlation report:
  - `python correlation_report.py`
- Train, save, and inspect logistic model:
  - `python logistic_risk_model.py --top 10 --retrain`
- Save coefficient drift metrics:
  - `python save_model_metrics.py`

## Logistic model math

- Features used:
  - `liquidity_depth`
  - `token_volatility`
  - `age_of_pool`
  - `volume_missing_penalty`
  - `il_penalty`
- Label:
  - `risk_event = 1` if next snapshot shows disappearance or a defined adverse move.
- Model:
  - `P(risk_event=1 | x) = sigmoid(beta0 + sum(beta_j * z_j))`
- Score used for ranking:
  - `risk_score = 1 - risk_probability`
  - `final_score = risk_score`

## Important behavioral details

- `python ai_yield_cli.py top --top 10` does not train coefficients.
- `python ai_yield_cli.py top --top 10 --model logit` does retrain coefficients every run.
- `save_model_metrics.py` also retrains and writes the saved logistic model plus CSV metrics.
- Saved logistic model path:
  - `data/logit_risk_model.pkl`
- Saved logistic metrics path:
  - `data/model_metrics_logit.csv`

## When modifying this project

- Keep Ethereum-only filtering unless explicitly asked to generalize.
- Keep the APY threshold unless explicitly asked to change it.
- If changing risk labels, update both `risk_model_utils.py` and any documentation that describes the label definition.
- If changing logistic persistence behavior, keep `top` behavior distinct from `dashboard`/`backtest` unless explicitly told otherwise.
