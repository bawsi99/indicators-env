"""
inference.py — Mandatory entry point for the Meta × PyTorch Hackathon automated evaluation.

Requirements:
1.  Reads API_BASE_URL, MODEL_NAME, and HF_TOKEN from environment variables.
2.  Uses the OpenAI Client for all LLM calls.
3.  Stdout follows strict [START], [STEP], and [END] logging format.
4.  Runtime < 20 min on 2 vCPU / 8 GB RAM.

IndicatorsEnv v3.0 — Portfolio MDP:
  The agent manages a long/short/flat position in a single NSE stock.
  Observation includes indicator snapshot + portfolio state (position, unrealized_pnl,
  capital_remaining) + macro context (Task 3 only).
  Actions: Bullish=Long, Bearish=Short, Neutral=Flat.
  Reward: actual next-day return × position − 0.1% transaction cost per trade change.
  Episode lengths: short=5 steps, medium=10 steps, long=20 steps.
"""

import os
import re
import json
import requests
import argparse
from openai import OpenAI

# --- Configuration (Mandatory per Hackathon Spec) ---
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1/")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN")

TASKS    = ["short_term_direction", "medium_term_direction", "long_term_conviction"]
ENV_TERMS = {"short_term_direction": "short", "medium_term_direction": "medium", "long_term_conviction": "long"}


def _parse_direction_and_conviction(text: str):
    """Extract direction and conviction from JSON-formatted LLM response."""
    # Clean possible markdown artifacts
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            d = str(obj.get("direction", "Neutral")).strip().capitalize()
            c = float(obj.get("conviction", 0.5))
            if d not in ("Bullish", "Bearish", "Neutral"):
                d = "Neutral"
            return d, max(0.0, min(1.0, c))
    except Exception:
        pass

    # Simple fallback: keyword search
    lower = text.lower()
    if "bullish" in lower: return "Bullish", 0.7
    if "bearish" in lower: return "Bearish", 0.7
    return "Neutral", 0.5


def run_evaluation(env_url: str, n_episodes: int) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_id in TASKS:
        term = ENV_TERMS[task_id]
        episode_results = []

        # [START] Log
        print(f"[START] task={task_id} env=IndicatorsEnv model={MODEL_NAME}", flush=True)

        for i in range(n_episodes):
            try:
                # 1. Reset — 60s timeout for cold-start yfinance fetch
                reset_resp = requests.post(
                    f"{env_url}/reset", params={"term": term}, timeout=60
                )
                if reset_resp.status_code != 200:
                    print(f"[ERR] episode={i+1} reset HTTP {reset_resp.status_code}", flush=True)
                    continue

                data       = reset_resp.json()
                obs        = data.get("observation", {})
                session_id = data.get("info", {}).get("session_id", "")
                max_steps  = data.get("info", {}).get("max_steps", 5)
                done       = False
                step_num   = 0
                prev_context = ""  # sequential feedback from previous step

                # 2. Multi-step loop — one episode = max_steps consecutive trading days
                while not done:
                    step_num += 1
                    symbol   = obs.get("symbol",           "N/A") if isinstance(obs, dict) else "N/A"
                    date     = obs.get("date",             "N/A") if isinstance(obs, dict) else "N/A"
                    position = obs.get("position",           0)   if isinstance(obs, dict) else 0
                    pnl      = obs.get("unrealized_pnl",   0.0)   if isinstance(obs, dict) else 0.0
                    capital  = obs.get("capital_remaining", 1.0)  if isinstance(obs, dict) else 1.0
                    macro    = obs.get("macro")                    if isinstance(obs, dict) else None

                    pos_label = {1: "Long", -1: "Short", 0: "Flat"}.get(position, "Flat")

                    # Build prompt — include portfolio state and macro context
                    system_prompt = (
                        "You are a quantitative portfolio manager trading a single NSE stock. "
                        "At each step you see technical indicators and your current portfolio state. "
                        "Choose a direction: Bullish (go/stay Long), Bearish (go/stay Short), "
                        "Neutral (exit/stay Flat). "
                        "Your reward is the actual next-day return × your position, minus 0.1% "
                        "transaction cost if you change position. Maximize cumulative return. "
                        'Respond ONLY with JSON: {"direction": "Bullish"|"Bearish"|"Neutral", '
                        '"conviction": <0.0-1.0>}'
                    )

                    macro_note = ""
                    if macro:
                        macro_note = (
                            f"\nMacro context: NIFTY50={macro.get('nifty_trend','?')} "
                            f"(20d return: {macro.get('nifty_return_20d',0):.1f}%), "
                            f"regime={macro.get('market_regime','?')}\n"
                        )

                    context_note = f"\nPrevious step: {prev_context}\n" if prev_context else ""

                    user_prompt = (
                        f"[Step {step_num}/{max_steps}] Stock: {symbol} | Date: {date} | "
                        f"Term: {term.upper()}\n"
                        f"Portfolio: position={pos_label}, unrealized_pnl={pnl:+.4f}, "
                        f"capital={capital:.4f}\n"
                        f"{macro_note}"
                        f"{context_note}"
                        f"Indicators: {obs.get('indicators', obs) if isinstance(obs, dict) else obs}\n\n"
                        f"Predict the {term}-term direction and manage your position."
                    )

                    # 3. Call LLM (using mandatory OpenAI Client)
                    chat_completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_prompt},
                        ],
                        max_tokens=128,
                        temperature=0.7,
                    )
                    response  = chat_completion.choices[0].message.content or ""
                    direction, conviction = _parse_direction_and_conviction(response)

                    # 4. Step Environment
                    step_resp = requests.post(
                        f"{env_url}/step",
                        params={"session_id": session_id},
                        json={"direction": direction, "conviction": conviction},
                        timeout=30,
                    )
                    if step_resp.status_code != 200:
                        print(f"[ERR] episode={i+1} step={step_num} HTTP {step_resp.status_code}", flush=True)
                        break

                    step_data    = step_resp.json()
                    reward       = step_data.get("reward", 0.0)
                    done         = step_data.get("done", True)
                    step_info    = step_data.get("info", {})
                    ground_truth = step_info.get("ground_truth", "N/A")
                    actual_ret   = step_info.get("actual_return_pct", 0.0)
                    new_capital  = step_info.get("capital", 1.0)
                    obs          = step_data.get("observation") or obs

                    # Build feedback for next step's prompt
                    correct_str  = "correct" if direction == ground_truth else "wrong"
                    prev_context = (
                        f"{direction} ({correct_str}, GT={ground_truth}), "
                        f"actual={actual_ret:+.2f}%, reward={reward:.4f}, capital={new_capital:.4f}"
                    )

                    # Track result for grader — key MUST be "predicted"
                    episode_results.append({
                        "predicted":    direction,
                        "conviction":   conviction,
                        "reward":       reward,
                        "ground_truth": ground_truth,
                    })

                    # [STEP] Log
                    print(
                        f"[STEP] step={step_num} action={direction} reward={reward:.4f} "
                        f"done={done} error=None",
                        flush=True,
                    )

            except Exception as e:
                print(f"[ERR] episode={i+1} {type(e).__name__}: {e}", flush=True)
                continue

        # 5. Finalize Task (Call Grader)
        try:
            grader_resp = requests.post(
                f"{env_url}/grader",
                json={"task_id": task_id, "episode_results": episode_results},
                timeout=60,
            )
            final_score = (
                grader_resp.json().get("score", 0.0)
                if grader_resp.status_code == 200 else 0.0
            )
        except Exception as e:
            print(f"[ERR] grader: {type(e).__name__}: {e}", flush=True)
            final_score = 0.0

        rewards_list = [ep["reward"] for ep in episode_results]
        steps_taken  = len(episode_results)
        success      = final_score > 0.0
        print(
            f"[END] success={success} steps={steps_taken} score={final_score:.4f} "
            f"rewards={rewards_list}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndicatorsEnv Portfolio MDP Inference Agent")
    parser.add_argument("--env_url",    default="http://localhost:7860", help="URL of the IndicatorsEnv server")
    parser.add_argument("--n_episodes", type=int, default=5,            help="Episodes per task (total 15)")
    args = parser.parse_args()

    # Mandatory Check: Environment Variables
    if not HF_TOKEN:
        print("Error: HF_TOKEN must be set as an environment variable.")
        exit(1)

    run_evaluation(args.env_url, args.n_episodes)
