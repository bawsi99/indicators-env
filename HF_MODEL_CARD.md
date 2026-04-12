---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- finance
- openenv
- grpo
- indicators-env
- technical-analysis
- relative-alpha
- market-neutral
- trl
- peft
- conversational
library_name: peft
pipeline_tag: text-generation
---

# IndicatorsEnv — GRPO Fine-tuned Qwen-7B (NSE India Relative Alpha)

This model is a **LoRA Adapter** for `Qwen2.5-7B-Instruct`, fine-tuned to solve
the **IndicatorsEnv v4.0** multi-stock relative alpha RL environment on real NSE (India) equities.

The agent observes 3 same-sector NSE stocks at each step and must identify which
stock has the strongest relative momentum signal, declare a direction, and calibrate
conviction. Reward = (chosen stock return − sector average) × direction × conviction × 50.

---

## Training Method

**Group Relative Policy Optimization (GRPO)** with QLoRA, run against IndicatorsEnv.

GRPO optimizes relative rewards within sampled trajectory groups — well-suited to the
Kelly conviction reward structure, where trajectories with high conviction on correct
picks score significantly higher than trajectories with low conviction or wrong picks.

---

## Training Results (Held-Out Evaluation, v3.0 Environment)

| Metric | Zero-Shot Baseline | GRPO Step 220 | Delta |
|---|---|---|---|
| Bullish Recall | 36% | 52% | **+16pp** |
| Bearish Recall | 0% | 20% | **+20pp** |
| Neutral Recall | 70% | 0% | −70pp |
| Macro F1 | 0.257 | 0.213 | −0.044 |

**Interpretation:** GRPO training shifted the model from Neutral-dominant behavior
(70% zero-shot Neutral recall) to active directional prediction. Bullish recall improved
+16pp and Bearish recall improved +20pp at peak checkpoint (step 220). The model
over-corrected: Neutral recall dropped to 0 across all 14 evaluated checkpoints
(steps 160–600), eliminating the Neutral class entirely.

This is a known GRPO failure mode — without explicit Neutral-class reward weighting,
the optimizer pushes the model toward the two classes with the highest reward variance
(Bullish and Bearish) and collapses Neutral to zero. Macro F1 reflects the precision
cost: 0.257 → 0.213.

**What this confirms:** The OpenEnv reward mechanism directly controls structural model
behavior at the class-recall level. The v4.0 market-neutral Kelly reward (where
overconfident wrong predictions lose proportionally) is designed to prevent this collapse
in the next training iteration.

---

## The Majority-Class Collapse Problem

Zero-shot LLMs on financial prediction tasks exhibit a systematic bias toward the
majority class. On NSE equities, "Neutral" (price moves within the threshold) is the
most frequent ground truth label — so models default to Neutral to minimize penalty.
This produces acceptable overall accuracy while completely failing to identify the
minority signals (Bullish/Bearish) that matter for trading.

GRPO training addressed the Neutral bias — Bullish and Bearish recall both improved
significantly — but introduced the opposite problem: Neutral was eliminated entirely.
True three-class balance requires explicit reward weighting for the Neutral class,
which is the target for the next training iteration.

---

## Usage (PEFT)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-7B-Instruct"
adapter_id    = "bawsi99/indicators-grpo-qwen7b"

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_id)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
```

The model expects a system prompt describing the multi-stock relative alpha task and
a user prompt containing the 3-stock indicator snapshot (from IndicatorsEnv v4.0).
It outputs JSON: `{"stock": "HDFCBANK", "direction": "Bullish", "conviction": 0.8}`

---

## Environment

- **IndicatorsEnv v4.0**: https://huggingface.co/spaces/bawsi99/indicators-env
- **Source Code**: https://github.com/bawsi99/indicators-env

---

*Created for the Meta × PyTorch Hackathon*
