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
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN")

TASKS = ["short_term_direction", "medium_term_direction", "long_term_direction"]
ENV_TERMS = {"short_term_direction": "short", "medium_term_direction": "medium", "long_term_direction": "long"}


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
        print(f"[START] task={task_id} n_episodes={n_episodes}")

        for i in range(n_episodes):
            try:
                # 1. Reset Environment
                reset_resp = requests.post(f"{env_url}/reset", json={"term": term}, timeout=30)
                if reset_resp.status_code != 200:
                    continue
                
                data = reset_resp.json()
                state = data.get("state", [])
                session_id = data.get("session_id", "")
                symbol = data.get("info", {}).get("symbol", "N/A")
                date = data.get("info", {}).get("date", "N/A")

                # 2. Build Prompt
                system_prompt = "You are a quantitative analyst. Given technical indicators, predict the directional price move as Bullish, Bearish, or Neutral. Provide reasoning then output a JSON with 'reasoning', 'direction', and 'conviction' (0.0-1.0)."
                user_prompt = f"Technical Indicators for {symbol} on {date}: {state}\n\nPredict the directional move."

                # 3. Call LLM (using mandatory OpenAI Client)
                chat_completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=256,
                    temperature=0.7
                )
                response = chat_completion.choices[0].message.content
                direction, conviction = _parse_direction_and_conviction(response)

                # 4. Step Environment
                step_resp = requests.post(
                    f"{env_url}/step", 
                    json={"action": direction, "session_id": session_id}, 
                    timeout=30
                )
                if step_resp.status_code != 200:
                    continue
                
                step_data = step_resp.json()
                reward = step_data.get("reward", 0.0)
                ground_truth = step_data.get("info", {}).get("ground_truth", "N/A")

                # Track result for grader
                episode_results.append({
                    "episode": i + 1,
                    "direction": direction,
                    "conviction": conviction,
                    "reward": reward,
                    "ground_truth": ground_truth
                })

                # [STEP] Log
                print(f"[STEP] episode={i+1} symbol={symbol} date={date} direction={direction} conviction={conviction} reward={reward:.4f} ground_truth={ground_truth}")

            except Exception as e:
                # Never crash the entire run
                continue

        # 5. Finalize Task (Call Grader)
        try:
            # Note: We provide the grader with the episode results for that task
            # In OpenEnv, the grader is usually a separate endpoint
            grader_resp = requests.post(
                f"{env_url}/grader", 
                json={"task": task_id, "results": episode_results},
                timeout=60
            )
            final_score = grader_resp.json().get("score", 0.0) if grader_resp.status_code == 200 else 0.0
            
            # [END] Log
            print(f"[END] task={task_id} score={final_score:.4f} episodes={len(episode_results)}")
        except:
            print(f"[END] task={task_id} score=0.0000 episodes={len(episode_results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict Compliance OpenEnv Inference Agent")
    parser.add_argument("--env_url", default="http://localhost:7860", help="URL of the IndicatorsEnv server")
    parser.add_argument("--n_episodes", type=int, default=5, help="Number of episodes per task (Total 15)")
    
    args = parser.parse_args()
    
    # Mandatory Check: Environment Variables
    if not API_BASE_URL or not HF_TOKEN:
        print("❌ Error: API_BASE_URL and HF_TOKEN must be set as environment variables.")
        exit(1)

    run_evaluation(args.env_url, args.n_episodes)
