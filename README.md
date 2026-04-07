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
license: mit
---

# IndicatorsEnv — OpenEnv RL Environment for Stock Market Technical Analysis

> **An OpenEnv-compatible RL environment where AI agents learn to interpret technical indicators and predict directional price moves on real NSE (Indian) stocks.**

---

## Environment Description

`IndicatorsEnv` simulates the task a quantitative analyst performs daily: given a snapshot of 25+ technical indicators for a real stock on a real historical date, predict the future price direction (Bullish / Bearish / Neutral) over a configurable forward window.

### Why This Matters
Technical indicator analysis is a genuine, high-value task performed by thousands of analysts globally. Training agents to reliably read indicators and predict directional moves has direct applications in:
- Automated trading systems
- Retail investment tools
- Portfolio risk management
- Agent-based financial research

### Data
- **Universe**: 50+ NSE (India National Stock Exchange) equities
- **Date Range**: 2020–2024 (5 years of real historical data)
- **Source**: `yfinance` API (free, reproducible)
- **Indicators**: 25+ indicators across 8 categories (see below)

---

## Action Space

```json
{
  "direction": "Bullish" | "Bearish" | "Neutral",
  "conviction": 0.0 to 1.0
}
```

| Field | Type | Description |
|---|---|---|
| `direction` | string (enum) | Predicted price direction over the forward window |
| `conviction` | float [0, 1] | Agent confidence. Higher = stronger signal. |

---

## Observation Space

```json
{
  "symbol": "RELIANCE",
  "date": "2023-04-12",
  "term": "MEDIUM",
  "current_price": 2456.75,
  "indicators": { ... }
}
```

### Indicator Suite (25+ signals across 8 categories)

| Category | Indicators |
|---|---|
| **Moving Averages** | SMA(20/50/200), EMA(20/50), Golden/Death Cross |
| **Momentum** | RSI(14) + signal, MACD(12/26/9) + crossover, Stochastic(14/3) |
| **Volatility** | Bollinger Bands (%, width, squeeze), ATR(14), volatility regime |
| **Trend** | ADX(14), +DI/-DI, trend strength |
| **Volume** | OBV, VWAP, MFI(14), CMF(20), A/D Line, volume ratio |
| **Levels** | Pivot Points (R2/R1/P/S1/S2) |

---

## Tasks

### Task 1: Short-term Direction *(Easy)*
- **Term**: 5-day forward return
- **Threshold**: ±1.0%
- **Grader**: Simple accuracy (fraction correct over N episodes)
- **Challenge**: High noise, large Neutral proportion. Model must avoid over-predicting Bullish.

### Task 2: Medium-term Direction *(Medium)*
- **Term**: 20-day forward return
- **Threshold**: ±2.5%
- **Grader**: Weighted accuracy — Bearish/Neutral correct predictions worth 1.5× (anti-majority-class bias)
- **Challenge**: Balanced class distribution requiring genuine indicator reading ability.

### Task 3: Long-term Conviction *(Hard)*
- **Term**: 60-day forward return
- **Threshold**: ±5.0%
- **Grader**: Direction *AND* conviction calibration. Correct + conviction ≥ 0.7 → 1.0. Correct + low conviction → 0.5. Wrong + high conviction → -0.1.
- **Challenge**: Requires both accurate prediction AND calibrated confidence — frontier-model difficulty.

---

## Reward Function

The reward is shaped over the full trajectory (not binary):

| Condition | Reward |
|---|---|
| Correct direction + conviction ≥ 0.6 | `1.0 + 0.1 × (conviction - 0.6)` (up to 1.04) |
| Correct direction + conviction < 0.6 | `1.0` |
| Wrong direction | `0.0` |
| Wrong + overconfident (conviction ≥ 0.8) | `-0.1` |

---

## Evaluation & Baseline Scores

| Task | Difficulty | Random Agent | OpenAI Baseline |
|---|---|---|---|
| Short-term Direction | Easy | ~0.33 | ~0.42 |
| Medium-term Direction | Medium | ~0.28 | ~0.46 |
| Long-term Conviction | Hard | ~0.20 | ~0.38 |

---

## Setup & Usage

### Local Setup
```bash
# 1. Clone and install
git clone https://github.com/bawsi99/indicators-env.git
cd indicators-env
pip install -r requirements.txt

# 2. Start the environment
cd env/
python indicators_env.py   # Runs on http://localhost:7860
```

### Docker
```bash
docker build -t indicators-env .
docker run -p 7860:7860 indicators-env
```

### Baseline Agent Inference
```bash
export API_BASE_URL=https://api-inference.huggingface.co/v1/
export MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
export HF_TOKEN=hf_your_token

python inference.py --env_url http://localhost:7860 --n_episodes 5
```

---

## Links

- **Hugging Face Space**: [bawsi99/indicators-env](https://huggingface.co/spaces/bawsi99/indicators-env)
- **GitHub Repository**: [bawsi99/indicators-env](https://github.com/bawsi99/indicators-env)
- **OpenEnv GitHub**: [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
