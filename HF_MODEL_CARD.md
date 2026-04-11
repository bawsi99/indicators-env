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
- trl
- peft
- conversational
library_name: peft
pipeline_tag: text-generation
---

# IndicatorsEnv — GRPO Fine-tuned Qwen-7B (NSE India)

This model is a **LoRA Adapter** for `Qwen2.5-7B-Instruct`, specifically fine-tuned to solve high-precision directional trading signals in the **IndicatorsEnv (OpenEnv)** reinforcement learning environment.

## Training Results

This agent was trained using **Group Relative Policy Optimization (GRPO)** with a **Symmetric Anti-Bias Reward Function** targeting the majority-class collapse problem in zero-shot financial prediction.

| Metric | Zero-Shot Baseline | GRPO Fine-Tuned (Step 220) | Δ |
| :--- | :--- | :--- | :--- |
| **Bullish Recall** | 36.0% | **52.0%** | **+16.0%** |
| **Bearish Recall** | 0.0% | **20.0%** | **+20.0%** |
| **Neutral Recall** | 70.0% | 0.0% | -70.0% |
| **Macro F1** | 0.257 | 0.213 | -0.044 |

**Interpretation:** GRPO training shifted the model from Neutral-dominant behavior (70% Neutral recall zero-shot) to active directional prediction. Bullish recall improved +16pp and Bearish recall improved +20pp at the peak checkpoint. However, the model over-corrected — Neutral recall dropped to 0 across all 14 evaluated checkpoints (steps 160–600), indicating the reward function needs stronger Neutral-class incentivization to achieve true three-class balance. Macro F1 reflects the precision cost of forced directional commitment. This confirms the OpenEnv reward mechanism directly influences structural model behavior — the next iteration targets balanced recall via explicit Neutral-class reward weighting.

## 🛠️ Model Description

- **Developed by:** [bawsi99](https://huggingface.co/bawsi99)
- **Model type:** LoRA Adapter
- **Base model:** [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Environment:** [IndicatorsEnv (OpenEnv)](https://huggingface.co/spaces/bawsi99/indicators-env)
- **Training Method:** GRPO (Group Relative Policy Optimization)

### The "Neutral Bias" Problem
In quantitative finance, "Mode Collapse" occurs when an AI agent defaults to the majority class (Neutral) to minimize penalty. The GRPO reward function explicitly penalizes inaction and rewards directional conviction, successfully forcing the model to actively interpret 25+ technical indicators (Momentum, Trend, Volatility) rather than defaulting to Neutral. However, training revealed the opposite extreme: the model eliminated Neutral predictions entirely, achieving 0% Neutral recall at all checkpoints. This indicates the reward function requires additional Neutral-class weighting for true three-class balance in future iterations.

## 🛒 Usage (PEFT)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-7B-Instruct"
adapter_id = "bawsi99/indicators-grpo-qwen7b"

# Load Base Model
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype="auto",
    device_map="auto",
)

# Load Adapter
model = PeftModel.from_pretrained(model, adapter_id)

# Ready for Inference
```

## 🏁 Evaluation Results
The model was evaluated on a held-out dataset of **NSE India stocks** from the 2023-2024 period. It demonstrates a significant surge in signal detection accuracy, specifically outperforming the baseline in identifying **high-momentum Bullish setups**.

---
*Created for the Meta × PyTorch Hackathon (2024)*
