#!/usr/bin/env python3
"""
baseline.py — Baseline inference script for IndicatorsEnv (OpenEnv Hackathon)

Uses the OpenAI API client to run a language model against all 3 IndicatorsEnv tasks.
The model receives a structured prompt of technical indicators and must output a
JSON prediction in the format: {"direction": "Bullish"|"Bearish"|"Neutral", "conviction": float}

Usage:
    # Against local environment server:
    python baseline.py --env_url http://localhost:7860

    # Against Hugging Face Space:
    python baseline.py --env_url https://bawsi99-indicators-env.hf.space

Environment variables:
    OPENAI_API_KEY   : Your OpenAI API key (required)
    OPENAI_BASE_URL  : Optional custom base URL (e.g. for local models or HF Inference API)
    OPENAI_MODEL     : Model to use (default: gpt-4o-mini)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── OpenAI Client Setup ──────────────────────────────────────────────────────

def _get_openai_client():
    """Initialize the OpenAI client from environment variables."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    base_url = os.environ.get("OPENAI_BASE_URL", None)
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client


# ─── Prompt Builder ───────────────────────────────────────────────────────────

def _build_prompt(observation: Dict[str, Any]) -> str:
    """Convert an IndicatorsEnv observation into a formatted model prompt."""
    ind = observation.get("indicators", {})
    ma  = ind.get("moving_averages", {})
    rsi = ind.get("rsi", {})
    mac = ind.get("macd", {})
    adx = ind.get("adx", {})
    vlt = ind.get("volatility", {})
    bb  = ind.get("bollinger_bands", {})
    vol = ind.get("enhanced_volume", {})
    t   = observation.get("term", "MEDIUM")

    prompt = f"""You are a quantitative analyst evaluating NSE (Indian) stocks.

Stock: {observation.get('symbol')} | Date: {observation.get('date')} | Price: {observation.get('current_price')}
Prediction Term: {t}

--- Technical Indicators ---
RSI(14): {rsi.get('rsi_14')} | RSI Signal: {rsi.get('rsi_signal')}
MACD Line: {mac.get('macd_line')} | Signal: {mac.get('signal_line')} | Histogram: {mac.get('histogram')}
ADX: {adx.get('adx')} | +DI: {adx.get('plus_di')} | -DI: {adx.get('minus_di')} | Trend: {adx.get('trend_strength')}
SMA20: {ma.get('sma_20')} | SMA50: {ma.get('sma_50')} | SMA200: {ma.get('sma_200')}
EMA20: {ma.get('ema_20')} | Cross: {ma.get('golden_cross')}
BB%: {bb.get('percent_b')} | BB Width: {bb.get('bandwidth')} | Squeeze: {bb.get('squeeze')}
ATR(14): {vlt.get('atr_14')} | Volatility Regime: {vlt.get('volatility_regime')}
VWAP: {vol.get('vwap')} | MFI: {vol.get('mfi')} | MFI Status: {vol.get('mfi_status')}
CMF: {vol.get('cmf')} | A/D Trend: {vol.get('ad_line_trend')}

Analyze these indicators and predict the {t.lower()}-term price direction.

Respond with ONLY valid JSON (no markdown, no explanation):
{{"direction": "Bullish" | "Bearish" | "Neutral", "conviction": <float 0.0-1.0>}}"""
    return prompt


# ─── Prediction Logic ─────────────────────────────────────────────────────────

def _parse_prediction(text: str) -> Tuple[str, float]:
    """Extract direction and conviction from model output."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            direction = str(obj.get("direction", "Neutral")).strip().capitalize()
            conviction = float(obj.get("conviction", 0.5))
            if direction not in ("Bullish", "Bearish", "Neutral"):
                direction = "Neutral"
            conviction = max(0.0, min(1.0, conviction))
            return direction, conviction
    except Exception:
        pass
    lower = text.lower()
    if "bullish" in lower:
        return "Bullish", 0.6
    if "bearish" in lower:
        return "Bearish", 0.6
    return "Neutral", 0.4


def _run_llm(client, model: str, prompt: str) -> Tuple[str, float]:
    """Call the LLM and parse the prediction."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a quantitative analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=64,
            temperature=0.1,
        )
        text = response.choices[0].message.content or ""
        return _parse_prediction(text)
    except Exception as e:
        logger.warning(f"[LLM] Call failed: {e}")
        return "Neutral", 0.5


# ─── Environment Client ───────────────────────────────────────────────────────

def _env_reset(env_url: str, term: str) -> Optional[Dict]:
    """Call /reset on the environment."""
    try:
        r = httpx.post(f"{env_url}/reset", params={"term": term}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"[Env] /reset failed: {e}")
        return None


def _env_step(env_url: str, session_id: str, direction: str, conviction: float) -> Optional[Dict]:
    """Call /step on the environment."""
    try:
        r = httpx.post(
            f"{env_url}/step",
            params={"session_id": session_id},
            json={"direction": direction, "conviction": conviction},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"[Env] /step failed: {e}")
        return None


def _env_grade(env_url: str, task_id: str, episode_results: List[Dict]) -> Optional[Dict]:
    """Call /grader on the environment."""
    try:
        r = httpx.post(
            f"{env_url}/grader",
            json={"task_id": task_id, "episode_results": episode_results},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"[Env] /grader failed: {e}")
        return None


# ─── Main Baseline Runner ─────────────────────────────────────────────────────

def run_baseline(env_url: str, n_episodes: int, model: str) -> None:
    client = _get_openai_client()

    # Get tasks from the environment
    try:
        tasks_resp = httpx.get(f"{env_url}/tasks", timeout=15)
        tasks_resp.raise_for_status()
        tasks = tasks_resp.json()["tasks"]
    except Exception as e:
        logger.error(f"[Env] /tasks failed: {e}")
        sys.exit(1)

    logger.info(f"[Baseline] Running {model} on {len(tasks)} tasks × {n_episodes} episodes each...")
    print(f"\n{'='*65}")
    print(f"BASELINE SCORES — IndicatorsEnv | Model: {model}")
    print(f"{'='*65}")

    all_scores = []

    for task in tasks:
        task_id = task["id"]
        term = task["term"]
        difficulty = task["difficulty"]

        logger.info(f"[{task_id}] Starting {n_episodes} episodes (term={term})...")
        episode_results = []
        total_reward = 0.0

        for ep in range(n_episodes):
            # 1. Reset
            reset_data = _env_reset(env_url, term=term)
            if reset_data is None:
                continue
            obs = reset_data["observation"]
            session_id = reset_data["info"]["session_id"]

            # 2. Build prompt and get LLM prediction
            prompt = _build_prompt(obs)
            direction, conviction = _run_llm(client, model, prompt)

            # 3. Step
            step_data = _env_step(env_url, session_id, direction, conviction)
            if step_data is None:
                continue

            reward = step_data.get("reward", 0.0)
            gt = step_data.get("info", {}).get("ground_truth", "")
            total_reward += reward

            episode_results.append({
                "ground_truth": gt,
                "predicted": direction,
                "conviction": conviction,
            })

            if (ep + 1) % 5 == 0:
                logger.info(f"[{task_id}] {ep+1}/{n_episodes} done | mean_reward={total_reward/(ep+1):.3f}")

        # 4. Grade
        grader_data = _env_grade(env_url, task_id, episode_results)
        score = grader_data["score"] if grader_data else 0.0
        all_scores.append(score)

        print(f"\nTask: {task['name']}")
        print(f"  Difficulty: {difficulty} | Episodes: {len(episode_results)}")
        print(f"  Mean Reward: {total_reward/max(1,len(episode_results)):.4f}")
        print(f"  Grader Score: {score:.4f}")

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n{'='*65}")
    print(f"OVERALL MEAN SCORE: {overall:.4f}")
    print(f"{'='*65}\n")

    # Save results
    out = {
        "model": model,
        "env_url": env_url,
        "n_episodes": n_episodes,
        "task_scores": dict(zip([t["id"] for t in tasks], all_scores)),
        "overall_mean": overall,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Results saved to baseline_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndicatorsEnv Baseline Inference Script")
    parser.add_argument("--env_url", default="http://localhost:7860", help="URL of the IndicatorsEnv server")
    parser.add_argument("--n_episodes", type=int, default=10, help="Episodes per task")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), help="OpenAI model to use")
    args = parser.parse_args()
    run_baseline(args.env_url, args.n_episodes, args.model)
