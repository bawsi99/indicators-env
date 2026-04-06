"""
eval_finetuned.py — Side-by-side evaluation of fine-tuned vs baseline models on IndicatorsEnv.

Compares:
  Baseline : Qwen2.5-1.5B-Instruct (zero-shot, no training)
  Fine-tuned: GRPO-trained Qwen2.5-1.5B with QLoRA adapter

Produces:
  - Overall accuracy (3-class)
  - Per-class Precision, Recall, F1
  - Full Confusion Matrix (before and after training)
  - Saved JSON with all metrics

Usage:
    python evaluation/eval_finetuned.py \\
        --adapter_path /content/drive/MyDrive/indicators_grpo_v2/lora_adapter \\
        --n_eval 100 \\
        --term medium
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Standard visualization imports
import matplotlib
matplotlib.use("Agg")  # Headless
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CLASSES = ["Bullish", "Bearish", "Neutral"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_direction(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            d = str(obj.get("direction", "Neutral")).strip().capitalize()
            return d if d in ("Bullish", "Bearish", "Neutral") else "Neutral"
    except Exception:
        pass
    lower = text.lower()
    if "bullish" in lower:
        return "Bullish"
    if "bearish" in lower:
        return "Bearish"
    return "Neutral"


def _full_metrics(results: List[Dict]) -> Dict:
    """Compute overall accuracy, per-class P/R/F1, and confusion matrix."""
    y_true = [r["ground_truth"] for r in results]
    y_pred = [r["predicted"] for r in results]

    # sklearn classification_report
    report = classification_report(
        y_true, y_pred,
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrix: rows=true, cols=predicted
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)

    # Per-class accuracy (recall)
    by_class_acc = {}
    for cls in CLASSES:
        cls_indices = [i for i, t in enumerate(y_true) if t == cls]
        if cls_indices:
            cls_correct = sum(1 for i in cls_indices if y_pred[i] == cls)
            by_class_acc[cls] = cls_correct / len(cls_indices)
        else:
            by_class_acc[cls] = 0.0

    return {
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "n_eval": len(results),
        "by_class_accuracy": by_class_acc,
        "precision": {cls: report[cls]["precision"] for cls in CLASSES},
        "recall": {cls: report[cls]["recall"] for cls in CLASSES},
        "f1_score": {cls: report[cls]["f1-score"] for cls in CLASSES},
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": CLASSES,
        "class_distribution": {cls: y_true.count(cls) for cls in CLASSES},
    }


def _print_full_report(label: str, metrics: Dict) -> None:
    print(f"\n{'─'*65}")
    print(f"  {label}")
    print(f"{'─'*65}")
    print(f"  Overall Accuracy : {metrics['overall_accuracy']:.4f}")
    print(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1      : {metrics['weighted_f1']:.4f}")
    print()
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'─'*52}")
    for cls in CLASSES:
        n = metrics["class_distribution"].get(cls, 0)
        print(
            f"  {cls:<10} "
            f"{metrics['precision'][cls]:>10.4f} "
            f"{metrics['recall'][cls]:>10.4f} "
            f"{metrics['f1_score'][cls]:>10.4f} "
            f"{n:>10}"
        )
    print()
    print("  Confusion Matrix (rows=Truth, cols=Predicted):")
    print(f"  {'':12}" + "".join(f"{cls:>10}" for cls in CLASSES))
    for i, cls in enumerate(CLASSES):
        row = metrics["confusion_matrix"][i]
        print(f"  {cls:<12}" + "".join(f"{v:>10}" for v in row))


def _sample_eval_episodes(n: int = 100, term: str = "medium") -> List[Dict]:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))
    from data_loader import generate_dataset_offline, NSE_UNIVERSE

    logger.info(f"[Eval] Sampling {n} eval episodes (offline, held-out 2023-2024)...")
    eval_symbols = NSE_UNIVERSE[30:50]
    raw = generate_dataset_offline(
        symbols=eval_symbols,
        start_date="2023-01-01",
        end_date="2024-06-30",
        term=term,
        dates_per_stock=max(1, n // len(eval_symbols) + 2),
        max_total=n,
    )

    episodes = []
    for item in raw:
        ind = item["indicators"]
        ma  = ind.get("moving_averages", {})
        rsi = ind.get("rsi", {})
        mac = ind.get("macd", {})
        adx = ind.get("adx", {})
        vlt = ind.get("volatility", {})
        bb  = ind.get("bollinger_bands", {})
        vol = ind.get("enhanced_volume", {})
        t   = item["term"]

        prompt = "\n".join([
            f"[TERM: {t}] Stock: {item['symbol']} | Date: {item['date']} | Price: {item['current_price']}",
            f"RSI(14)={rsi.get('rsi_14')} | RSI_Signal={rsi.get('rsi_signal')}",
            f"MACD={mac.get('macd_line')} | Signal={mac.get('signal_line')} | Histogram={mac.get('histogram')}",
            f"ADX={adx.get('adx')} | +DI={adx.get('plus_di')} | -DI={adx.get('minus_di')}",
            f"SMA20={ma.get('sma_20')} | SMA50={ma.get('sma_50')} | EMA20={ma.get('ema_20')}",
            f"BB%={bb.get('percent_b')} | BB_Width={bb.get('bandwidth')} | Squeeze={bb.get('squeeze')}",
            f"ATR={vlt.get('atr_14')} | Regime={vlt.get('volatility_regime')}",
            f"VWAP={vol.get('vwap')} | MFI={vol.get('mfi')} | CMF={vol.get('cmf')}",
            f"Predict the {t}-term price direction. Respond ONLY with valid JSON:",
            '{"direction": "Bullish" | "Bearish" | "Neutral", "conviction": <float 0-1>}',
        ])
        episodes.append({"prompt": prompt, "ground_truth": item["ground_truth"]})

    logger.info(f"[Eval] Sampled {len(episodes)} eval episodes")
    return episodes


def _run_model(model, tokenizer, episodes: List[Dict], label: str) -> List[Dict]:
    results = []
    model.eval()
    for i, ep in enumerate(episodes):
        try:
            inputs = tokenizer(ep["prompt"], return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            predicted = _parse_direction(completion)
        except Exception as e:
            logger.warning(f"[{label}] Episode {i} failed: {e}")
            predicted = "Neutral"

        results.append({"ground_truth": ep["ground_truth"], "predicted": predicted})
        if (i + 1) % 10 == 0:
            logger.info(f"[{label}] {i+1}/{len(episodes)} done")
    return results


# ─── Plotting Helpers ────────────────────────────────────────────────────────

def _load_trainer_state(checkpoint_dir: Path) -> List[Dict[str, Any]]:
    """Scan checkpoint subdirectories and collect log history."""
    all_logs = []
    seen_steps = set()

    if not checkpoint_dir.exists():
        return []

    # Look for checkpoint-* folders
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]) if "-" in d.name else 0
    )

    for ckpt in checkpoints:
        state_file = ckpt / "trainer_state.json"
        if not state_file.exists():
            continue

        with open(state_file) as f:
            state = json.load(f)

        for entry in state.get("log_history", []):
            step = entry.get("step")
            if step is not None and step not in seen_steps:
                seen_steps.add(step)
                all_logs.append(entry)

    all_logs.sort(key=lambda x: x.get("step", 0))
    return all_logs


def _extract_series(logs: List[Dict], key: str):
    steps, values = [], []
    for entry in logs:
        if key in entry and "step" in entry:
            steps.append(entry["step"])
            values.append(entry[key])
    return steps, values


def _plot_training_curves(adapter_path: str) -> str:
    """Generate and save loss/reward charts. Returns path to saved png.
    Skips gracefully for Hugging Face Hub IDs (no local trainer_state.json).
    """
    # Detect HF Hub IDs (e.g. "bawsi99/indicators-grpo-qwen1.5b") vs local paths
    is_hub_id = "/" in adapter_path and not os.path.isabs(adapter_path) and not os.path.exists(adapter_path)
    if is_hub_id:
        logger.info("[Plot] Skipping charts for remote Hub ID (no local trainer_state.json).")
        return ""
    checkpoint_dir = Path(adapter_path).parent
    logger.info(f"[Plot] Scanning for logs in: {checkpoint_dir}")
    
    logs = _load_trainer_state(checkpoint_dir)
    if not logs:
        logger.warning("[Plot] No trainer logs found. Skipping charts.")
        return ""

    reward_steps, reward_vals = _extract_series(logs, "reward")
    if not reward_vals:
         reward_steps, reward_vals = _extract_series(logs, "rewards/mean")
    loss_steps, loss_vals = _extract_series(logs, "loss")
    
    n_plots = sum([bool(reward_vals), bool(loss_vals)])
    if n_plots == 0:
        logger.warning("[Plot] No 'reward' or 'loss' keys found in log history.")
        return ""

    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 5))
    if n_plots == 1: axes = [axes]

    fig.suptitle("IndicatorsEnv — GRPO Training Curves", fontsize=13, fontweight="bold")

    ax_idx = 0
    if reward_vals:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(reward_steps, reward_vals, color="#4CAF50", alpha=0.3, label="Per-step")
        if len(reward_vals) >= 5:
            window = 10
            rolled = [statistics.mean(reward_vals[max(0, i-window//2):i+window//2+1]) for i in range(len(reward_vals))]
            ax.plot(reward_steps, rolled, color="#1B5E20", linewidth=2, label=f"Smooth (w={window})")
        ax.set_title("Reward Trajectory"); ax.legend(); ax.grid(True, alpha=0.2)

    if loss_vals:
        ax = axes[ax_idx]
        ax.plot(loss_steps, loss_vals, color="#2196F3", label="Training Loss")
        ax.set_title("Optimization Loss"); ax.legend(); ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = Path(adapter_path) / "training_curves.png"
    plt.savefig(out_path, dpi=150)
    logger.info(f"[Plot] Charts saved to {out_path}")
    return str(out_path)


# ─── Main ─────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    from peft import PeftModel
    from transformers import BitsAndBytesConfig

    episodes = _sample_eval_episodes(n=args.n_eval, term=args.term)
    if not episodes:
        logger.error("[Eval] No eval episodes. Check data_loader.")
        sys.exit(1)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    # ── Resolve the correct base model by probing actual weight shapes ────────
    # adapter_config.json can be incorrect (e.g., checkpoint-160 has 7B weights
    # but the config says 1.5B). Safetensors never lies — read q_proj.lora_A to
    # infer the real hidden dimension and select the matching base model.
    is_hub_id = "/" in args.adapter_path and not os.path.isabs(args.adapter_path) and not os.path.exists(args.adapter_path)
    resolved_model_id = MODEL_ID  # Default

    def _detect_model_from_weights(adapter_path: str) -> str:
        """Read one lora_A tensor to infer hidden size -> base model."""
        try:
            from safetensors import safe_open
            weights_file = Path(adapter_path) / "adapter_model.safetensors"
            if not weights_file.exists():
                return None
            with safe_open(str(weights_file), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "lora_A" in key and "q_proj" in key:
                        input_dim = f.get_tensor(key).shape[-1]
                        if input_dim >= 3500:   # Qwen2.5-7B hidden_size=3584
                            return "Qwen/Qwen2.5-7B-Instruct"
                        elif input_dim >= 2000:  # Qwen2.5-3B hidden_size=2048
                            return "Qwen/Qwen2.5-3B-Instruct"
                        else:                    # Qwen2.5-1.5B hidden_size=1536
                            return "Qwen/Qwen2.5-1.5B-Instruct"
        except Exception as e:
            logger.warning(f"[Eval] Tensor probe failed: {e}")
        return None

    if not is_hub_id:
        detected = _detect_model_from_weights(args.adapter_path)
        if detected:
            resolved_model_id = detected
            logger.info(f"[Eval] 🔍 Auto-detected base model from weights: {resolved_model_id}")
        else:
            # Fallback: read adapter_config.json
            try:
                cfg_path = Path(args.adapter_path) / "adapter_config.json"
                with open(cfg_path) as f:
                    adapter_cfg = json.load(f)
                resolved_model_id = adapter_cfg.get("base_model_name_or_path", MODEL_ID)
                logger.info(f"[Eval] 📄 Base model from adapter_config.json: {resolved_model_id}")
            except Exception as e:
                logger.warning(f"[Eval] Could not detect base model, using default {MODEL_ID}: {e}")
    else:
        # HF Hub ID: fetch adapter_config.json from the Hub
        try:
            import urllib.request
            cfg_url = f"https://huggingface.co/{args.adapter_path}/raw/main/adapter_config.json"
            with urllib.request.urlopen(cfg_url) as resp:
                adapter_cfg = json.loads(resp.read())
            resolved_model_id = adapter_cfg.get("base_model_name_or_path", MODEL_ID)
            logger.info(f"[Eval] ☁️  Base model from HF Hub adapter_config: {resolved_model_id}")
        except Exception as e:
            logger.warning(f"[Eval] Could not fetch HF Hub adapter_config: {e}. Using {MODEL_ID}")

    if resolved_model_id != MODEL_ID:
        logger.warning(f"[Eval] ⚠️  Using {resolved_model_id} (not default {MODEL_ID})")

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Baseline: Zero-shot ───────────────────────────────────────────────────
    baseline_metrics = None
    if not args.skip_baseline:
        logger.info(f"[Eval] Loading baseline model (zero-shot): {resolved_model_id}...")
        dtype = torch.float16 if args.low_memory else torch.bfloat16
        base_model = AutoModelForCausalLM.from_pretrained(
            resolved_model_id, 
            quantization_config=bnb, 
            device_map="auto",
            trust_remote_code=True, 
            torch_dtype=dtype,
            low_cpu_mem_usage=args.low_memory,
        )
        baseline_results = _run_model(base_model, tokenizer, episodes, "Baseline")
        baseline_metrics = _full_metrics(baseline_results)
    else:
        logger.info("[Eval] Skipping baseline computation (using --skip_baseline)")

    # ── Fine-tuned: GRPO + QLoRA ─────────────────────────────────────────────
    logger.info(f"[Eval] Loading fine-tuned adapter from {args.adapter_path}...")
    if args.skip_baseline:
        # We need to load the base model first since baseline was skipped
        dtype = torch.float16 if args.low_memory else torch.bfloat16
        base_model = AutoModelForCausalLM.from_pretrained(
            resolved_model_id, 
            quantization_config=bnb, 
            device_map="auto",
            trust_remote_code=True, 
            torch_dtype=dtype,
            low_cpu_mem_usage=args.low_memory,
        )
    
    # Load adapter onto base model
    ft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    
    ft_results = _run_model(ft_model, tokenizer, episodes, "FineTuned(GRPO)")
    ft_metrics = _full_metrics(ft_results)

    # ── Full Report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"EVALUATION RESULTS — IndicatorsEnv ({args.term} term) | N={args.n_eval}")
    print("=" * 65)

    if baseline_metrics:
        _print_full_report("Zero-shot Qwen2.5-7B (Baseline)", baseline_metrics)
    
    _print_full_report("GRPO Fine-tuned Qwen2.5-7B (Ours)", ft_metrics)

    if baseline_metrics:
        delta_acc = ft_metrics["overall_accuracy"] - baseline_metrics["overall_accuracy"]
        delta_f1  = ft_metrics["macro_f1"] - baseline_metrics["macro_f1"]

        print(f"\n{'='*65}")
        print(f"  Δ Overall Accuracy : {delta_acc:+.4f} ({'✅ improved' if delta_acc > 0 else '❌ degraded'})")
        print(f"  Δ Macro F1         : {delta_f1:+.4f} ({'✅ improved' if delta_f1 > 0 else '❌ degraded'})")
        print(f"  Bearish Recall Δ   : {ft_metrics['recall']['Bearish'] - baseline_metrics['recall']['Bearish']:+.4f}")
        print(f"  Neutral Recall Δ   : {ft_metrics['recall']['Neutral'] - baseline_metrics['recall']['Neutral']:+.4f}")
        print(f"{'='*65}\n")
    else:
        delta_acc = 0; delta_f1 = 0

    # ── Save Results ─────────────────────────────────────────────────────────
    out = {
        "model_id": MODEL_ID,
        "adapter_path": str(args.adapter_path),
        "n_eval": args.n_eval,
        "term": args.term,
        "baseline_zeroshot": baseline_metrics,
        "finetuned_grpo": ft_metrics,
        "delta_accuracy": delta_acc,
        "delta_macro_f1": delta_f1,
    }
    out_path = Path(args.adapter_path) / "eval_results_full.json"
    # For Hugging Face Hub IDs, save to a local folder named after the repo slug
    is_hub_id = "/" in args.adapter_path and not os.path.isabs(args.adapter_path) and not os.path.exists(args.adapter_path)
    if is_hub_id:
        local_slug = args.adapter_path.split("/")[-1]
        out_dir = Path(local_slug)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "eval_results_full.json"
    else:
        out_path = Path(args.adapter_path) / "eval_results_full.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"[Eval] Full results saved to {out_path}")

    # ── 6. Generate Training Charts ──────────────────────────────────────────
    _plot_training_curves(args.adapter_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full evaluation: fine-tuned vs zero-shot baseline")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct", help="Base model ID")
    parser.add_argument("--adapter_path", required=True, help="Path to saved LoRA adapter")
    parser.add_argument("--n_eval", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--term", default="medium", help="Prediction term: short|medium|long")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip the baseline zero-shot evaluation")
    parser.add_argument("--low_memory", action="store_true", help="Optimize for low VRAM (float16 + low_cpu_mem_usage)")
    args = parser.parse_args()
    
    # Run evaluation with the specified model
    evaluate(args)
