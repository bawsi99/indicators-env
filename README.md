---
title: IndicatorsEnv
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - finance
  - stock-market
  - technical-analysis
  - rl-environment
  - multi-stock
  - market-neutral
license: mit
---

# IndicatorsEnv v4.0 — Multi-stock Relative Alpha MDP

> **An OpenEnv-compatible RL environment where AI agents learn to pick the strongest
> relative momentum stock within a sector — and profit from within-sector alpha,
> not broad market beta.**

---

## Environment Description

`IndicatorsEnv v4.0` is a **multi-stock, market-neutral** RL environment built on real
NSE (India) equity data.

At each step the agent observes **3 stocks from the same NSE sector** — all with full
technical indicator snapshots — and must decide:

1. **Which stock** has the most actionable signal? (or pass with NONE)
2. **Bullish or Bearish** on that stock relative to the sector?
3. **How confident?** (conviction = Kelly fraction wagered)

**Reward = (chosen stock's period return − sector average) × direction × conviction × 50**

This is a **market-neutral** formulation: the sector average cancels broad market beta,
so a random policy earns ≈ 0 expected reward. A skilled policy earns consistently
positive alpha by correctly identifying within-sector outperformers — a genuine,
learnable signal documented in the empirical finance literature (within-sector
momentum, RSI divergence, MACD crossover relative strength).

### Why This Design

| Property | v4.0 (this) | Naive single-stock env |
|---|---|---|
| **MDP or bandit?** | Genuine MDP — stock selection creates causal opportunity cost in sequential signal history | Contextual bandit — each step is independent |
| **Reward learnability** | Random policy ≈ 0; skill earns positive alpha | Random policy ≈ 0.33 on 3-class problem (inflated baseline) |
| **GT overlap** | Zero — step spacing = GT window (reward measures the exact return you predicted) | 95% overlap at 10 daily steps × 20-day GT window |
| **Calibration incentive** | Kelly conviction: wrong+high conviction loses; correct+high conviction wins | Binary correct/wrong |
| **Selective participation** | NONE pass option — agent learns when to engage vs. sit out | Forced picks on every step |

---

## Action Space

```json
{"stock": "HDFCBANK", "direction": "Bullish", "conviction": 0.8}
```
```json
{"stock": "NONE", "direction": "NONE", "conviction": 0.0}
```

| Field | Type | Values | Description |
|---|---|---|---|
| `stock` | string | NSE symbol or `"NONE"` | Stock to trade from the 3 available, or pass |
| `direction` | string | `Bullish` / `Bearish` / `NONE` | Long / short alpha bet, or skip |
| `conviction` | float | 0.0 – 1.0 | Kelly fraction: higher = larger bet |

---

## Observation Space

```json
{
  "step": 3,
  "max_steps": 10,
  "term": "MEDIUM",
  "sector": "banking",
  "available_stocks": ["HDFCBANK", "ICICIBANK", "AXISBANK"],
  "stocks": {
    "HDFCBANK":  {"current_price": 1650.5, "rsi_14": 62.3, "rsi_trend": "up",   "price_momentum_pct": 2.4,  "indicators": {...}},
    "ICICIBANK": {"current_price": 1012.0, "rsi_14": 48.1, "rsi_trend": "down", "price_momentum_pct": -0.8, "indicators": {...}},
    "AXISBANK":  {"current_price": 1123.5, "rsi_14": 55.7, "rsi_trend": "up",   "price_momentum_pct": 1.1,  "indicators": {...}}
  },
  "signal_history": [
    {"step": 1, "picked_stock": "HDFCBANK", "direction": "Bullish", "ground_truth": "Bullish", "correct": true,  "alpha_pct": 1.8, "reward": 0.63},
    {"step": 2, "picked_stock": "NONE",     "direction": "NONE",    "ground_truth": "N/A",     "correct": null,  "alpha_pct": 0.0, "reward": 0.0}
  ],
  "macro": null
}
```

### Per-stock Indicator Suite (25+ signals, 8 categories)

| Category | Indicators |
|---|---|
| **Moving Averages** | SMA(20/50/200), EMA(20/50), Golden/Death Cross, MA signal |
| **Momentum** | RSI(14) + trend + status, MACD(12/26/9) + crossover signal |
| **Volatility** | Bollinger Bands (%, width, squeeze), ATR(14), volatility regime |
| **Trend** | ADX(14), +DI/-DI, trend strength |
| **Volume** | OBV, VWAP, MFI(14), CMF(20), A/D Line, volume ratio |
| **Stochastic** | %K(14), %D(3), signal |
| **Levels** | Pivot Points (R2/R1/P/S1/S2) |

### Signal History

`signal_history` accumulates across all steps of an episode — showing which stocks
were picked, whether the prediction was correct, and what alpha was earned. This
gives the agent genuine sequential state: RSI divergence trends, multi-week momentum
shifts, and selective participation patterns are all visible.

---

## Tasks

### Task 1: Short-term Relative Alpha *(Easy)*
- **Steps**: 5 (1 trading day apart → 1 week)
- **GT window**: 1-day return, threshold ±0.3%
- **Grader**: Directional accuracy on active (non-NONE) steps
- **Challenge**: Daily noise is high; agent must identify the one stock with a clean signal vs. the other two

### Task 2: Medium-term Relative Alpha *(Medium)*
- **Steps**: 10 (5 trading days apart → 10 weekly observations)
- **GT window**: 5-day return, threshold ±1.5%
- **Grader**: Weighted accuracy (Bearish/Neutral correct = 1.5×) + participation rate bonus
- **Challenge**: Weekly signals require reading multi-indicator confluence; signal history spans 2.5 months

### Task 3: Long-term Risk-Constrained Alpha *(Hard)*
- **Steps**: 15 (20 trading days apart → 15 monthly observations)
- **GT window**: 20-day return, threshold ±2.5%
- **Grader**: Conviction-calibrated (correct+conviction≥0.7 = 1.0; wrong+conviction≥0.8 = −0.1; normalized to (0,1))
- **Extras**: Macro context (NIFTY50 trend, market regime) in observation; episode ends early if drawdown > 10%
- **Challenge**: 15-month episode spans 2–3 full market regimes; macro context + sector momentum combine for genuine multi-horizon reasoning

---

## Reward Function

```
alpha  = chosen_stock_period_return - mean(all_3_stocks_period_return)
reward = alpha × direction_sign × conviction × 50
```

| Scenario | Expected reward |
|---|---|
| Correct pick (stock outperforms), Bullish, conviction=0.7, alpha=+2% | +0.70 |
| Wrong pick (stock underperforms), Bullish, conviction=0.7, alpha=−2% | −0.70 |
| Correct pick, Bearish, conviction=0.5, alpha=−3% | +0.75 |
| NONE (pass) | 0.00 |
| Random policy (expected) | ≈ 0.00 |

Reward range: [−1.5, 1.5]. NONE steps yield 0.0 and are excluded from grader scoring.

---

## Episode Structure

**Same 3 stocks** are observed throughout one episode, with step spacing creating
genuine temporal distance between observations.

```
Episode Example (medium task, sector=banking):
  reset()   → HDFCBANK, ICICIBANK, AXISBANK | starting 2022-09-05
  step  1   → Week of 2022-09-05 | pick HDFCBANK → Bullish 0.8 | alpha=+1.8% | reward=+0.72 | done=False
  step  2   → Week of 2022-09-12 | NONE (no clear signal)       | reward= 0.00 | done=False
  step  3   → Week of 2022-09-19 | pick AXISBANK → Bearish 0.6  | alpha=+0.9% | reward=+0.27 | done=False
  ...
  step 10   → Week of 2022-11-21 | pick ICICIBANK → Bullish 0.7 | alpha=+2.1% | reward=+0.74 | done=True
```

Inference log (what the evaluation script produces):

```
[START] task=medium_term_direction env=IndicatorsEnv model=meta-llama/Llama-3.2-1B-Instruct
[STEP]  step=1  stock=HDFCBANK  action=Bullish  reward=0.7200  done=False  error=None
[STEP]  step=2  stock=NONE      action=NONE     reward=0.0000  done=False  error=None
[STEP]  step=3  stock=AXISBANK  action=Bearish  reward=0.2700  done=False  error=None
[STEP]  step=4  stock=HDFCBANK  action=Bullish  reward=-0.3500 done=False  error=None
[STEP]  step=5  stock=NONE      action=NONE     reward=0.0000  done=False  error=None
[STEP]  step=6  stock=ICICIBANK action=Bullish  reward=0.5100  done=False  error=None
[STEP]  step=7  stock=AXISBANK  action=Bearish  reward=0.1800  done=False  error=None
[STEP]  step=8  stock=NONE      action=NONE     reward=0.0000  done=False  error=None
[STEP]  step=9  stock=HDFCBANK  action=Bullish  reward=0.6300  done=False  error=None
[STEP]  step=10 stock=ICICIBANK action=Bullish  reward=0.7400  done=True   error=None
[END]   success=True steps=10 score=0.6200 rewards=[0.72, 0.0, 0.27, -0.35, 0.0, 0.51, 0.18, 0.0, 0.63, 0.74]
```

---

## Baseline Scores

| Task | Random Agent | Expected Skilled Agent |
|---|---|---|
| Short-term Relative Alpha | ≈ 0.33 (active steps) | > 0.50 |
| Medium-term Relative Alpha | ≈ 0.33 | > 0.50 |
| Long-term Risk-Constrained | ≈ 0.09 (conviction-calibrated) | > 0.40 |

The random agent NONE pass rate is ~33% (1 in 3 actions); active steps score ≈ 0.33
accuracy on a balanced 3-class GT distribution.

---

## GRPO Fine-tuning Results (Qwen2.5-7B)

GRPO training was run against an earlier version of this environment to demonstrate
that the reward mechanism directly alters structural model behavior.

| Metric | Zero-Shot | GRPO Step 220 | Delta |
|---|---|---|---|
| Bullish Recall | 36% | 52% | +16pp |
| Bearish Recall | 0% | 20% | +20pp |
| Neutral Recall | 70% | 0% | −70pp |
| Macro F1 | 0.257 | 0.213 | −0.044 |

**Interpretation:** GRPO shifted the model from Neutral-dominant behavior (70% zero-shot
Neutral recall) to active directional prediction: Bullish recall +16pp, Bearish recall
+20pp at peak checkpoint (step 220). However, the model over-corrected — Neutral recall
dropped to 0 across all 14 evaluated checkpoints (steps 160–600). This confirms the
OpenEnv reward mechanism directly influences structural model behavior. The over-correction
indicates the reward function needs stronger Neutral-class incentivization to achieve
true three-class balance. The v4.0 market-neutral reward design addresses this:
Kelly conviction penalizes overconfident wrong predictions, which should prevent the
forced-commitment collapse observed in v3.0 training.

---

## Setup & Usage

### Local

```bash
# Install dependencies
pip install fastapi uvicorn yfinance pandas numpy

# Start the environment server
cd env/
python indicators_env.py   # FastAPI on http://localhost:7860

# Check health
curl http://localhost:7860/health
```

### Docker

```bash
docker build -t indicators-env .
docker run -p 7860:7860 indicators-env
```

### Run Inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1/
export MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct
export HF_TOKEN=hf_your_token

python inference.py --env_url http://localhost:7860 --n_episodes 5
```

### Quick API Test

```bash
# Reset — picks a random sector and 3 stocks
curl -s -X POST "http://localhost:7860/reset?term=short" | python3 -m json.tool

# Step — pick HDFCBANK, bet Bullish with conviction 0.7
curl -s -X POST "http://localhost:7860/step?session_id=<ID>" \
  -H "Content-Type: application/json" \
  -d '{"stock": "HDFCBANK", "direction": "Bullish", "conviction": 0.7}' | python3 -m json.tool

# Pass a step
curl -s -X POST "http://localhost:7860/step?session_id=<ID>" \
  -H "Content-Type: application/json" \
  -d '{"stock": "NONE", "direction": "NONE", "conviction": 0.0}' | python3 -m json.tool

# Grader (active steps only)
curl -s -X POST "http://localhost:7860/grader" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "short_term_direction", "episode_results": [
        {"predicted": "Bullish", "ground_truth": "Bullish", "conviction": 0.7},
        {"predicted": "NONE",    "ground_truth": "N/A",     "conviction": 0.0},
        {"predicted": "Bearish", "ground_truth": "Bearish", "conviction": 0.8}
      ]}' | python3 -m json.tool
```

---

## Links

- **Hugging Face Space**: [bawsi99/indicators-env](https://huggingface.co/spaces/bawsi99/indicators-env)
- **GitHub Repository**: [bawsi99/indicators-env](https://github.com/bawsi99/indicators-env)
- **OpenEnv GitHub**: [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
