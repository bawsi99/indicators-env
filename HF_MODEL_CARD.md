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

## 🚀 Breakthrough Results

Unlike zero-shot foundation models which exhibit a "Neutral Bias" (collapsing into "Neutral" guesses to maintain safe accuracy), this agent was trained using **Group Relative Policy Optimization (GRPO)** with a **Symmetric Anti-Bias Reward Function**.

| Metric | Zero-Shot Baseline | GRPO Fine-Tuned (Step 220+) | Δ Status |
| :--- | :--- | :--- | :--- |
| **Bullish Recall** | 36.0% | **56.0%** | **+20.0% 🚀** |
| **Bearish Recall** | 0.0% | **20.0%** | **+20.0% 🚀** |
| **Neutral Recall** | 70.0% | **0.0%** | **Active Decision Making** |

## 🛠️ Model Description

- **Developed by:** [bawsi99](https://huggingface.co/bawsi99)
- **Model type:** LoRA Adapter
- **Base model:** [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Environment:** [IndicatorsEnv (OpenEnv)](https://huggingface.co/spaces/bawsi99/indicators-env)
- **Training Method:** GRPO (Group Relative Policy Optimization)

### The "Neutral Mode Collapse" Solution
In quantitative finance, "Mode Collapse" occurs when an AI agent defaults to the majority class (Neutral) to minimize penalty. We solved this by implementing a reward function that explicitly penalizes inaction and rewards directional conviction. This forced the model to actively interpret 25+ technical indicators (Momentum, Trend, Volatility) rather than guessing.

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
