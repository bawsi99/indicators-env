import argparse
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import GRPOConfig, GRPOTrainer

# Version Stamp for User Confirmation
print("### VERIFIED VERSION: 1.0.7 ###")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add env/ to path for offline data loader
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'env'))
from data_loader import generate_dataset_offline, NSE_UNIVERSE

# ─── Configuration ──────────────────────────────────────────────────────────

# Core model
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# ─── Inline Reward Function ──────────────────────────────────────────────────

def inline_reward_fn(completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
    """
    Symmetric reward function to prevent class-bias collapse.
    - Matches correct direction: +2.0
    - Mismatches direction: -1.0
    - Format bonus (valid JSON): +1.0
    - Rare class bonus (Bearish/Neutral): +0.5
    """
    rewards = []
    for completion, expected in zip(completions, ground_truth):
        reward = 0.0
        try:
            # 1. Format Reward (+1.0)
            match = re.search(r'"direction":\s*"([^"]+)"', completion, re.IGNORECASE)
            if match:
                reward += 1.0  # Valid JSON-like format bonus
                predicted = match.group(1).capitalize()
            else:
                predicted = "Invalid"
            
            # 2. Accuracy Reward (+2.0 balance)
            if predicted == expected.capitalize():
                reward += 2.0
                if expected.capitalize() != "Bullish":
                    reward += 0.5  # Slight discovery bonus for hard classes
            elif predicted == "Invalid":
                reward -= 1.0  # Penalize failing to follow format
            else:
                reward -= 1.0  # standard penalty for wrong guess
                
        except Exception as e:
            logger.warning(f"[Reward] Parse failed: {e}")
            reward -= 1.0
        
        rewards.append(reward)
    return rewards


# ─── Training logic ──────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    logger.info(f"[Train] Starting offline build for {args.num_episodes} episodes...")

    # Slice NSE_UNIVERSE for symbols
    target_symbols = NSE_UNIVERSE[:args.symbols_count]

    # 1. Build Offline Dataset
    raw_items = generate_dataset_offline(
        symbols=target_symbols,
        dates_per_stock=args.dates_per_stock,
        max_total=args.num_episodes,
        term=args.term,
    )
    
    if not raw_items:
        logger.error("[Train] FAILED to build dataset. Check internet/yfinance.")
        return

    # Convert to Prompt + Ground Truth format for GRPO
    dataset_data = []
    for item in raw_items:
        ind = item["indicators"]
        ma  = ind.get("moving_averages", {})
        rsi = ind.get("rsi", {})
        mac = ind.get("macd", {})
        adx = ind.get("adx", {})
        vlt = ind.get("volatility", {})
        t   = item["term"]

        # Minimalist prompt for faster iteration
        prompt = "\n".join([
            f"[TERM: {t}] Stock: {item['symbol']} | Date: {item['date']} | Price: {item['current_price']}",
            f"RSI={rsi.get('rsi_14')} | MACD={mac.get('macd_line')} | ADX={adx.get('adx')}",
            f"SMA20={ma.get('sma_20')} | SMA50={ma.get('sma_50')} | ATR={vlt.get('atr_14')}",
            f"Predict {t}-term direction. Respond ONLY:",
            '{"direction": "Bullish" | "Bearish" | "Neutral", "conviction": <float>}',
        ])
        
        dataset_data.append({
            "prompt": prompt,
            "ground_truth": item["ground_truth"],
        })

    full_ds = Dataset.from_list(dataset_data)
    split = full_ds.train_test_split(test_size=0.1)
    train_ds = split["train"]
    eval_ds = split["test"]
    logger.info(f"[Train] Dataset ready: {len(train_ds)} train, {len(eval_ds)} eval.")

    # 2. Setup Model (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"[Train] Loading base model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. GRPO Config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 16 // args.batch_size), # Target effective batch ~128 (8 gens * 16 steps)
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=5,
        save_steps=20,
        bf16=True,
        gradient_checkpointing=True,
        max_steps=args.max_steps,
        max_completion_length=32,
        num_generations=8,  # Increased from 4 for better GRPO advantage variance
        report_to="none",
    )

    # 4. GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        reward_funcs=[inline_reward_fn],
    )

    logger.info(f"[Train] Starting GRPO training (resume={args.resume_from_checkpoint})...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 5. Save final LoRA adapter
    output_path = Path(args.output_dir) / "lora_adapter"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info(f"[Train] Training complete. LoRA adapter saved to {output_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO fine-tune Qwen2.5-7B on IndicatorsEnv")
    parser.add_argument("--env_url",    default="http://localhost:8000", help="Archived: Env URL (not used in offline mode)")
    parser.add_argument("--model_id",   default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", default="./outputs/indicators_grpo")
    parser.add_argument("--num_episodes",    type=int, default=1000)
    parser.add_argument("--symbols_count",   type=int, default=30)
    parser.add_argument("--dates_per_stock", type=int, default=40)
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--lora_r",       type=int, default=16)
    parser.add_argument("--epochs",       type=int, default=1)
    parser.add_argument("--max_steps",    type=int, default=-1, help="Stop after this many steps (useful for resuming)")
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--term",         type=str.lower, default="medium", help="intraday, short, medium, or long")
    parser.add_argument("--resume_from_checkpoint", default=None, help="Path to checkpoint folder to resume from")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
