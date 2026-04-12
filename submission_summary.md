# Hackathon Project Report: IndicatorsEnv v4.0 (OpenEnv)

---

## 1. Context: The Meta × PyTorch Hackathon

The **Meta × PyTorch Hackathon** challenges developers to build a real-world Reinforcement Learning (RL) environment using the **OpenEnv** standard.

The goal is to create a fully standardized, containerized API — exposing `step()`, `reset()`, `state()`, `grader()`, and `baseline()` endpoints — that simulates a **genuine real-world task** where agents can receive feedback and learn over episodes.

---

## 2. My Project: IndicatorsEnv v4.0 — Multi-stock Relative Alpha MDP

Most participants build toy logic problems or text-adventure games. I built a genuinely hard problem: **within-sector relative momentum prediction** on real NSE (India) equities.

**IndicatorsEnv v4.0** is a multi-stock, market-neutral RL environment. At each step the agent observes **3 stocks from the same NSE sector** and must:

1. Pick which stock has the strongest relative momentum signal (or pass with NONE)
2. Declare Bullish (long alpha) or Bearish (short alpha)
3. Set conviction — the Kelly fraction of virtual wealth wagered

**Reward = (chosen stock period return − sector average) × direction × conviction × 50**

This is a **market-neutral** design: the sector average cancels broad market beta, so a random policy earns ≈ 0 expected reward. A skilled policy earns positive alpha by correctly identifying within-sector outperformers.

### Why This Is a Genuine MDP (Not a Contextual Bandit)

| Property | IndicatorsEnv v4.0 | Naive single-stock env |
|---|---|---|
| **Sequential state** | Signal history accumulates: RSI trends, price momentum, past picks all visible in observation | Each step is stateless |
| **Opportunity cost** | Picking RELIANCE while HDFC runs +3% vs sector is a real causal loss | No stock choice, no opportunity cost |
| **GT overlap** | Zero — step spacing = GT window (reward measures the exact return predicted) | Up to 95% overlap at daily steps × 20-day GT |
| **Calibration incentive** | Kelly conviction: overconfident wrong predictions lose proportionally | Binary correct/wrong |
| **Selective participation** | NONE pass — agent learns when setups exist vs. when to sit out | Forced prediction every step |

### Why NSE India

NSE is a semi-strong form efficient market at best — retail participation is high, institutional research coverage of mid/small-cap stocks is thin, and within-sector momentum effects are well-documented in emerging market equity literature. This means a learnable signal exists, making it a better RL training ground than a fully efficient market where no agent could learn anything.

---

## 3. Environment Design

### Three Tasks of Increasing Difficulty

| Task | Steps | Spacing | GT Window | Difficulty |
|---|---|---|---|---|
| Short-term Relative Alpha | 5 | 1 day | 1-day return ±0.3% | Easy |
| Medium-term Relative Alpha | 10 | 5 days | 5-day return ±1.5% | Medium |
| Long-term Risk-Constrained Alpha | 15 | 20 days | 20-day return ±2.5% | Hard |

Step spacing = GT window for every task. This guarantees zero ground-truth overlap between consecutive steps — the reward signal measures exactly the return period the agent predicted.

### Task 3 Extras
- **Macro context**: NIFTY50 trend, 20-day return, market regime (trending/ranging) added to each observation
- **Drawdown limit**: Episode terminates early if virtual capital drawdown exceeds 10%
- **15-month span**: Monthly steps across multiple market regimes test genuine multi-horizon reasoning

### Observation: Signal History
Each step's observation includes a `signal_history` list that grows across the episode — showing which stocks were picked, directions, ground truth correctness, alpha earned, and RSI snapshots. This creates genuine sequential state: the agent can detect multi-week RSI divergence trends, accumulated momentum signals, and its own track record within the episode.

### Data Pipeline
- **Source**: `yfinance` — real NSE OHLCV data, reproducible, no API key required
- **Indicators**: 25+ signals across 8 categories (Moving Averages, RSI, MACD, Bollinger Bands, ADX, Stochastic, Volume suite, Pivot Points)
- **Episode construction**: Single yfinance call per stock (3 per episode) fetches the full window — no per-step API calls

---

## 4. GRPO Training Results

The hackathon only requires building the environment and running a baseline evaluation. I went further: I ran a real **GRPO fine-tuning loop** on `Qwen2.5-7B-Instruct` to demonstrate that the reward mechanism directly alters structural model behavior.

### Results (Held-Out Episodes, QLoRA GRPO)

| Metric | Zero-Shot | GRPO Step 220 | Delta |
|---|---|---|---|
| Bullish Recall | 36% | 52% | +16pp |
| Bearish Recall | 0% | 20% | +20pp |
| Neutral Recall | 70% | 0% | −70pp |
| Macro F1 | 0.257 | 0.213 | −0.044 |

**What the training shows:** GRPO shifted the model from Neutral-dominant behavior (70% Neutral recall zero-shot) to active directional prediction — Bullish recall +16pp, Bearish recall +20pp at peak (step 220). The model over-corrected: Neutral recall dropped to 0 across all 14 evaluated checkpoints (steps 160–600), reflecting the precision cost of forced directional commitment. Macro F1 moved 0.257 → 0.213.

**What this proves:** The OpenEnv reward mechanism directly controls structural model behavior. The over-correction is a known GRPO failure mode when minority class reward weighting is absent. The v4.0 Kelly conviction reward directly addresses this: overconfident wrong predictions lose proportionally, which should discourage the forced-commitment collapse observed in v3.0 training.

See `final_evaluation_curve.png` for the full checkpoint trajectory.

---

## 5. Reproducibility

All artifacts are publicly available:

| Artifact | Location |
|---|---|
| Live HF Space (Environment) | https://huggingface.co/spaces/bawsi99/indicators-env |
| Fine-tuned LoRA Weights | https://huggingface.co/bawsi99/indicators-grpo-qwen7b |
| Source Code | `hackathon/` directory |

### Reproduce Baseline Evaluation Locally

```bash
# Start the environment server
cd hackathon/env && python indicators_env.py &

# Run inference against the live server
export API_BASE_URL=https://router.huggingface.co/v1/
export MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct
export HF_TOKEN=hf_your_token

python hackathon/inference.py --env_url http://localhost:7860 --n_episodes 3
```

### Expected Output

```
[START] task=short_term_direction env=IndicatorsEnv model=meta-llama/Llama-3.2-1B-Instruct
[STEP]  step=1 stock=HDFCBANK  action=Bullish  reward=0.6800 done=False error=None
[STEP]  step=2 stock=NONE      action=NONE     reward=0.0000 done=False error=None
[STEP]  step=3 stock=ICICIBANK action=Bearish  reward=0.3100 done=False error=None
[STEP]  step=4 stock=HDFCBANK  action=Bullish  reward=-0.2400 done=False error=None
[STEP]  step=5 stock=AXISBANK  action=Bullish  reward=0.5200 done=True  error=None
[END]   success=True steps=5 score=0.6000 rewards=[0.68, 0.0, 0.31, -0.24, 0.52]
```

---

## 6. Status

Environment is live, all three tasks are implemented and graded, baseline runs correctly, documentation is complete. Fully ready for Phase 3 review.
