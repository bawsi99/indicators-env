"""
inference.py — Mandatory entry point for the Meta × PyTorch Hackathon automated evaluation.

Requirements:
1.  Reads API_BASE_URL, MODEL_NAME, and HF_TOKEN from environment variables.
2.  Uses the OpenAI Client for all LLM calls.
3.  Stdout follows strict [START], [STEP], and [END] logging format.
4.  Runtime < 20 min on 2 vCPU / 8 GB RAM.
"""

import os
import re
import json
import requests
import argparse
from openai import OpenAI

# --- Configuration (Mandatory per Hackathon Spec) ---
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")

TASKS = ["short_term_direction", "medium_term_direction", "long_term_conviction"]
ENV_TERMS = {"short_term_direction": "short", "medium_term_direction": "medium", "long_term_conviction": "long"}


def _parse_direction_and_conviction(text: str):
    """Extract direction and conviction from JSON-formatted LLM response."""
    # Clean possible markdown artifacts
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        # Look for JSON object
        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            d = str(obj.get("direction", "Neutral")).strip().capitalize()
            c = float(obj.get("conviction", 0.5))
            if d not in ("Bullish", "Bearish", "Neutral"): d = "Neutral"
            return d, c
    except Exception:
        pass
    
    # Simple fallback: keyword search
    lower = text.lower()
    if "bullish" in lower: return "Bullish", 0.7
    if "bearish" in lower: return "Bearish", 0.7
    return "Neutral", 0.5


def run_evaluation(env_url, n_episodes):
    # Initialize OpenAI Client
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_id in TASKS:
        term = ENV_TERMS[task_id]
        episode_results = []

        # [START] Log
        print(f"[START] task={task_id} env=IndicatorsEnv model={MODEL_NAME}", flush=True)

        for i in range(n_episodes):
            try:
                # 1. Reset Environment — 60s timeout for cold-start yfinance fetch
                reset_resp = requests.post(f"{env_url}/reset", params={"term": term}, timeout=60)
                if reset_resp.status_code != 200:
                    continue

                data = reset_resp.json()
                obs = data.get("observation", {})
                session_id = data.get("info", {}).get("session_id", "")
                done = False
                step_num = 0
                prev_context = ""  # feedback from previous step for sequential prompting

                # 2. Multi-step loop: one episode = 5 consecutive trading days
                while not done:
                    step_num += 1
                    symbol = obs.get("symbol", "N/A") if isinstance(obs, dict) else "N/A"
                    date   = obs.get("date",   "N/A") if isinstance(obs, dict) else "N/A"

                    # Build Prompt — include previous step result at steps 2+
                    system_prompt = (
                        "You are a quantitative analyst. Given technical indicators, "
                        "predict the directional price move as Bullish, Bearish, or Neutral. "
                        "Respond with JSON: {\"direction\": \"Bullish\"|\"Bearish\"|\"Neutral\", \"conviction\": <0.0-1.0>}"
                    )
                    context_note = f"\nPrevious step result: {prev_context}\n" if prev_context else ""
                    user_prompt = (
                        f"[Step {step_num}/5] Stock: {symbol} | Date: {date}\n"
                        f"{context_note}"
                        f"Indicators: {obs}\n\nPredict the {term}-term direction."
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
                    response = chat_completion.choices[0].message.content or ""
                    direction, conviction = _parse_direction_and_conviction(response)

                    # 4. Step Environment
                    step_resp = requests.post(
                        f"{env_url}/step",
                        params={"session_id": session_id},
                        json={"direction": direction, "conviction": conviction},
                        timeout=30,
                    )
                    if step_resp.status_code != 200:
                        break

                    step_data    = step_resp.json()
                    reward       = step_data.get("reward", 0.0)
                    done         = step_data.get("done", True)
                    ground_truth = step_data.get("info", {}).get("ground_truth", "N/A")
                    obs          = step_data.get("observation") or obs

                    # Update context for next step
                    correct_str  = "correct" if direction == ground_truth else "wrong"
                    prev_context = f"predicted {direction}, actual was {ground_truth} ({correct_str}), reward {reward:.2f}"

                    # Track result for grader — key MUST be "predicted" (not "direction")
                    episode_results.append({
                        "predicted":    direction,
                        "conviction":   conviction,
                        "reward":       reward,
                        "ground_truth": ground_truth,
                    })

                    # [STEP] Log — done reflects actual value, NOT hardcoded True
                    print(f"[STEP] step={step_num} action={direction} reward={reward:.4f} done={done} error=None", flush=True)

            except Exception:
                # Never crash the entire run
                continue

        # 5. Finalize Task (Call Grader)
        try:
            grader_resp = requests.post(
                f"{env_url}/grader",
                json={"task_id": task_id, "episode_results": episode_results},
                timeout=60,
            )
            final_score = grader_resp.json().get("score", 0.0) if grader_resp.status_code == 200 else 0.0

            # [END] Log
            rewards_list = [ep["reward"] for ep in episode_results]
            steps_taken  = len(episode_results)
            success      = final_score > 0.0
            print(f"[END] success={success} steps={steps_taken} score={final_score:.4f} rewards={rewards_list}", flush=True)
        except Exception:
            rewards_list = [ep["reward"] for ep in episode_results]
            steps_taken  = len(episode_results)
            print(f"[END] success=False steps={steps_taken} score=0.0000 rewards={rewards_list}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict Compliance OpenEnv Inference Agent")
    parser.add_argument("--env_url", default="http://localhost:7860", help="URL of the IndicatorsEnv server")
    parser.add_argument("--n_episodes", type=int, default=5, help="Number of episodes per task (Total 15)")
    
    args = parser.parse_args()
    
    # Mandatory Check: Environment Variables
    if not HF_TOKEN:
        print("❌ Error: HF_TOKEN must be set as an environment variable.")
        exit(1)

    run_evaluation(args.env_url, args.n_episodes)
