"""
inference.py — Mandatory entry point for the Meta × PyTorch Hackathon automated evaluation.

Requirements:
1. Reads API_BASE_URL, MODEL_NAME, and HF_TOKEN from environment variables.
2. Uses the OpenAI Client for all LLM calls.
3. Stdout follows strict [START], [STEP], and [END] logging format.
4. Runtime < 20 min on 2 vCPU / 8 GB RAM.

IndicatorsEnv v4.1 — Multi-stock Portfolio MDP:
  At each step the agent observes 3 stocks from the same NSE sector.
  It picks ONE stock and declares Bullish/Bearish, or passes with NONE.
  Reward = (chosen_stock_return − sector_avg) × direction × conviction × 50
  Market-neutral: random policy earns ~0; skilled policy earns positive alpha.
  Tasks: short (5 steps, daily), medium (10 steps, weekly), long (15 steps, monthly).
"""

import os
import re
import json
import requests
import argparse
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI

# ── Configuration (Mandatory per Hackathon Spec) ────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1/")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.2-1B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN")

TASKS     = ["short_term_direction", "medium_term_direction", "long_term_conviction"]
ENV_TERMS = {
    "short_term_direction": "short",
    "medium_term_direction": "medium",
    "long_term_conviction": "long",
}


# ── Action parser ────────────────────────────────────────────────────────────

def _parse_action(
    text: str,
    available_stocks: List[str],
) -> Tuple[str, str, float]:
    """
    Parse LLM response into (stock, direction, conviction).

    Expected JSON:
      {"stock": "HDFCBANK", "direction": "Bullish", "conviction": 0.8}
    or to pass:
      {"stock": "NONE", "direction": "NONE", "conviction": 0.0}

    Falls back gracefully on malformed output.
    """
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()

    try:
        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if match:
            obj = json.loads(match.group(0))
            stock     = str(obj.get("stock", "NONE")).strip().upper()
            direction = str(obj.get("direction", "NONE")).strip().capitalize()
            conviction = float(obj.get("conviction", 0.5))
            conviction = max(0.0, min(1.0, conviction))

            # Validate stock
            if stock not in available_stocks and stock != "NONE":
                # Try case-insensitive match
                upper_map = {s.upper(): s for s in available_stocks}
                stock = upper_map.get(stock, "NONE")

            # Validate direction
            if direction not in ("Bullish", "Bearish", "None"):
                direction = "NONE"
            if stock == "NONE":
                direction = "NONE"

            return stock, direction, conviction
    except Exception:
        pass

    # ── Fallback: keyword search ─────────────────────────────────────────────
    lower = text.lower()

    # Look for stock mention
    found_stock = "NONE"
    for sym in available_stocks:
        if sym.lower() in lower:
            found_stock = sym
            break

    # Look for direction
    if "bullish" in lower:
        direction = "Bullish"
    elif "bearish" in lower:
        direction = "Bearish"
    elif "none" in lower or "pass" in lower or "skip" in lower:
        direction = "NONE"
        found_stock = "NONE"
    else:
        direction = "NONE"
        found_stock = "NONE"

    return found_stock, direction, 0.6


# ── Prompt builder ───────────────────────────────────────────────────────────

def _build_prompt(
    obs: Dict[str, Any],
    step_num: int,
    max_steps: int,
    term: str,
    task_id: str,
    signal_history: List[Dict[str, Any]],
    prev_info: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Build system + user prompt for multi-stock relative alpha prediction.
    """
    system_prompt = (
        "You are a quantitative portfolio manager specializing in NSE (India) equities. "
        "At each step you observe 3 stocks from the SAME sector. "
        "Your job: identify which stock has the strongest relative momentum signal "
        "and bet on its direction vs. the sector average. "
        "Reward = (your stock's return − sector average) × direction × conviction × 50. "
        "Market-neutral: picking randomly earns ~0 reward — you profit ONLY by selecting "
        "the outperformer correctly. "
        "Switching your held position to a different stock costs 0.1% of capital in transaction cost. "
        "If you are already in a strong position, staying in the same stock avoids switching cost. "
        "Pass with NONE to hold your position without incurring a switch cost. "
        'Respond ONLY with JSON: {"stock": "<SYMBOL>", "direction": "Bullish"|"Bearish", '
        '"conviction": <0.0-1.0>} or {"stock": "NONE", "direction": "NONE", "conviction": 0.0}'
    )

    sector      = obs.get("sector", "unknown")
    available   = obs.get("available_stocks", [])
    stocks_data = obs.get("stocks", {})
    macro       = obs.get("macro")

    # ── Format stock snapshots ────────────────────────────────────────────────
    stock_lines = []
    for sym in available:
        s = stocks_data.get(sym, {})
        rsi       = s.get("rsi_14", "N/A")
        rsi_trend = s.get("rsi_trend", "?")
        momentum  = s.get("price_momentum_pct", 0.0)
        price     = s.get("current_price", "N/A")
        # Key indicator signals
        inds      = s.get("indicators", {})
        macd_sig  = inds.get("macd", {}).get("signal", "?")
        ma_sig    = inds.get("moving_averages", {}).get("signal", "?")
        adx_str   = inds.get("adx", {}).get("trend_strength", "?")

        stock_lines.append(
            f"  {sym}: price={price}  RSI={rsi}({rsi_trend})  "
            f"momentum={momentum:+.1f}%  MACD={macd_sig}  MA={ma_sig}  ADX={adx_str}"
        )

    stocks_str = "\n".join(stock_lines)

    # ── Signal history (last 3 steps for context) ────────────────────────────
    history_str = ""
    if signal_history:
        recent = signal_history[-3:]
        lines = []
        for h in recent:
            picked = h.get("picked_stock", "NONE")
            dirn   = h.get("direction", "NONE")
            gt     = h.get("ground_truth", "?")
            alpha  = h.get("alpha_pct", 0.0)
            rew    = h.get("reward", 0.0)
            if dirn != "NONE":
                correct_str = "✓" if h.get("correct") else "✗"
                lines.append(
                    f"  Step {h['step']}: picked {picked} → {dirn} "
                    f"({correct_str} GT={gt}  alpha={alpha:+.2f}%  reward={rew:.3f})"
                )
            else:
                lines.append(f"  Step {h['step']}: NONE (passed)")
        history_str = "Signal history:\n" + "\n".join(lines) + "\n"

    # ── Macro context (Task 3) ────────────────────────────────────────────────
    macro_str = ""
    if macro:
        macro_str = (
            f"Macro: NIFTY50={macro.get('nifty_trend','?')} "
            f"(20d return: {macro.get('nifty_return_20d',0):.1f}%)  "
            f"regime={macro.get('market_regime','?')}\n"
        )

    # ── Previous step feedback ────────────────────────────────────────────────
    prev_str = ""
    if prev_info and prev_info.get("chosen_stock") != "N/A":
        prev_str = (
            f"Last step: picked {prev_info.get('chosen_stock','?')} → "
            f"{prev_info.get('direction','?')} "
            f"(GT={prev_info.get('ground_truth','?')}  "
            f"alpha={prev_info.get('alpha_pct',0):+.2f}%  "
            f"reward={prev_info.get('reward',0):.3f})\n"
        )

    # ── Portfolio context (v4.1) ──────────────────────────────────────────────
    holding  = obs.get("current_holding", "NONE") if isinstance(obs, dict) else "NONE"
    capital  = obs.get("capital", 1.0)             if isinstance(obs, dict) else 1.0
    drawdown = obs.get("drawdown", 0.0)            if isinstance(obs, dict) else 0.0
    if holding != "NONE":
        tx_note = f"(switching from {holding} costs ~{capital * 0.001:.4f} tx_cost)"
    else:
        tx_note = "(no current holding — first pick is free)"
    portfolio_str = (
        f"Portfolio: capital={capital:.4f}  holding={holding}  "
        f"drawdown={drawdown:.1%}  {tx_note}\n"
    )

    user_prompt = (
        f"[Step {step_num}/{max_steps}] Sector: {sector.upper()} | Task: {task_id} | "
        f"Term: {term.upper()}\n\n"
        f"Stocks (same sector):\n{stocks_str}\n\n"
        f"{history_str}"
        f"{macro_str}"
        f"{portfolio_str}"
        f"{prev_str}"
        f"Pick the stock with the strongest relative momentum signal.\n"
        f"Available: {available} or NONE to skip.\n\n"
        f"Which stock has the best setup? Respond with JSON."
    )

    return system_prompt, user_prompt


# ── Main evaluation loop ────────────────────────────────────────────────────

def run_evaluation(env_url: str, n_episodes: int) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_id in TASKS:
        term = ENV_TERMS[task_id]
        episode_results: List[Dict[str, Any]] = []

        print(f"[START] task={task_id} env=IndicatorsEnv model={MODEL_NAME}", flush=True)

        for i in range(n_episodes):
            try:
                # 1. Reset — 90s timeout (multi-stock fetches 3 yfinance calls)
                reset_resp = requests.post(
                    f"{env_url}/reset",
                    params={"term": term},
                    timeout=90,
                )
                if reset_resp.status_code != 200:
                    print(f"[ERR] episode={i+1} reset HTTP {reset_resp.status_code}", flush=True)
                    continue

                data       = reset_resp.json()
                obs        = data.get("observation", {})
                session_id = data.get("info", {}).get("session_id", "")
                max_steps  = data.get("info", {}).get("max_steps", 5)
                available  = obs.get("available_stocks", [])
                done       = False
                step_num   = 0
                prev_info: Optional[Dict[str, Any]] = None

                # 2. Multi-step loop — one episode = max_steps steps
                while not done:
                    step_num += 1

                    # Signal history from observation (accumulated across steps)
                    signal_history = obs.get("signal_history", []) if isinstance(obs, dict) else []
                    available      = obs.get("available_stocks", available) if isinstance(obs, dict) else available

                    # Build prompts
                    system_prompt, user_prompt = _build_prompt(
                        obs           = obs if isinstance(obs, dict) else {},
                        step_num      = step_num,
                        max_steps     = max_steps,
                        term          = term,
                        task_id       = task_id,
                        signal_history= signal_history,
                        prev_info     = prev_info,
                    )

                    # 3. LLM call
                    chat_completion = client.chat.completions.create(
                        model    = MODEL_NAME,
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_prompt},
                        ],
                        max_tokens  = 128,
                        temperature = 0.7,
                    )
                    response  = chat_completion.choices[0].message.content or ""
                    stock, direction, conviction = _parse_action(response, available)

                    # 4. Step environment
                    step_resp = requests.post(
                        f"{env_url}/step",
                        params={"session_id": session_id},
                        json={
                            "stock":     stock,
                            "direction": direction,
                            "conviction": conviction,
                        },
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
                    obs          = step_data.get("observation") or obs

                    # Store info for next step's prompt
                    prev_info = {
                        "chosen_stock":      step_info.get("chosen_stock", stock),
                        "direction":         direction,
                        "ground_truth":      ground_truth,
                        "alpha_pct":         step_info.get("alpha_pct", 0.0),
                        "reward":            reward,
                    }

                    # Track for grader — key MUST be "predicted"
                    episode_results.append({
                        "predicted":    direction,
                        "conviction":   conviction,
                        "reward":       reward,
                        "ground_truth": ground_truth,
                    })

                    # [STEP] log
                    print(
                        f"[STEP] step={step_num} stock={stock} action={direction} "
                        f"reward={reward:.4f} done={done} error=None",
                        flush=True,
                    )

            except Exception as e:
                print(f"[ERR] episode={i+1} {type(e).__name__}: {e}", flush=True)
                continue

        # 5. Finalize task — call grader
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
    parser = argparse.ArgumentParser(description="IndicatorsEnv v4.0 Inference Agent")
    parser.add_argument("--env_url",    default="http://localhost:7860", help="IndicatorsEnv server URL")
    parser.add_argument("--n_episodes", type=int, default=5,            help="Episodes per task")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("Error: HF_TOKEN must be set as an environment variable.")
        exit(1)

    run_evaluation(args.env_url, args.n_episodes)
