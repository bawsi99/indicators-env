"""
indicators_env.py — OpenEnv-compatible RL environment for NSE equity portfolio management.

Implements the full OpenEnv standard interface:
  - HTTP endpoints: reset / step / state / tasks / grader / baseline / health
  - WebSocket (/ws) for low-latency sequential step() calls during RL training

Env spec (v3.0):
  Observation : Full indicator snapshot + portfolio state (position, unrealized_pnl,
                capital_remaining) + macro context (Task 3 only)
  Action      : {"direction": "Bullish"|"Bearish"|"Neutral", "conviction": float 0-1}
                Bullish = enter/hold Long (+1), Bearish = enter/hold Short (-1),
                Neutral = exit/stay Flat (0)
  Reward      : actual_next_day_return × position − 0.1% transaction_cost_if_changed
                Scaled ×50 so a 2% daily return = reward ≈ 1.0

Tasks (v3.0):
  Task 1 (Easy)   : SHORT  — 5 steps (1 week),  ±1.5% threshold
  Task 2 (Medium) : MEDIUM — 10 steps (2 weeks), ±2.5% threshold, transaction costs
  Task 3 (Hard)   : LONG   — 20 steps (4 weeks), ±5.0% threshold,
                             drawdown limit 5%, macro context in observation
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from data_loader import (
    NSE_UNIVERSE,
    build_observation,
    build_multi_step_episode,
    compute_ground_truth,
    generate_scenario_pool,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="IndicatorsEnv",
    description=(
        "OpenEnv-compatible RL environment for NSE equity portfolio management. "
        "An AI agent receives technical indicator snapshots and manages a long/short/flat "
        "position across a multi-step episode. Reward is driven by actual market returns "
        "minus transaction costs. Three tasks of increasing difficulty: "
        "5-step weekly trading (easy), 10-step two-week position management (medium), "
        "20-step monthly trading with drawdown constraint and macro context (hard). "
        "Built for the Meta × PyTorch Hackathon."
    ),
    version="3.0.0",
)

# ─── Environment constants ────────────────────────────────────────────────────

# Episode length per task — genuine difficulty ladder
TASK_MAX_STEPS: Dict[str, int] = {
    "short":  5,   # 1 trading week
    "medium": 10,  # 2 trading weeks
    "long":   20,  # 4 trading weeks
}

TRANSACTION_COST = 0.001   # 0.1% per trade (realistic NSE brokerage + STT)
DRAWDOWN_LIMIT   = 0.05    # 5% max drawdown; Task 3 terminates early if exceeded
REWARD_SCALE     = 50.0    # ×50: a 2% daily return → reward ≈ 1.0

# ─── Task Definitions ─────────────────────────────────────────────────────────

TASKS = [
    {
        "id": "short_term_direction",
        "name": "Short-term Position Management (Easy)",
        "description": (
            "Manage a long/short/flat position in a single NSE stock over 5 consecutive "
            "trading days (one week). The agent receives full technical indicator snapshots "
            "and its current portfolio state (position, unrealized P&L, capital). "
            "Reward = actual next-day return × position − 0.1% transaction cost per trade. "
            "Actions: Bullish=Long, Bearish=Short, Neutral=Flat. "
            "Ground-truth label: 5-day forward return threshold ±1.5%."
        ),
        "difficulty": "easy",
        "term": "short",
        "episode_steps": TASK_MAX_STEPS["short"],
        "grader_note": "Score = fraction of correct directional predictions over all steps.",
        "action_schema": {
            "direction": {"type": "string", "enum": ["Bullish", "Bearish", "Neutral"]},
            "conviction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
    {
        "id": "medium_term_direction",
        "name": "Medium-term Position Management (Medium)",
        "description": (
            "Manage a position over 10 consecutive trading days (two weeks). "
            "Transaction costs (0.1% per position change) create a genuine "
            "exploration-exploitation tradeoff — the agent must decide when to hold "
            "vs. flip vs. exit. Portfolio state (position, P&L) is visible in observation. "
            "Reward = actual next-day return × position − transaction cost. "
            "Ground-truth label: 20-day forward return threshold ±2.5%."
        ),
        "difficulty": "medium",
        "term": "medium",
        "episode_steps": TASK_MAX_STEPS["medium"],
        "grader_note": (
            "Score = weighted accuracy: Bearish/Neutral correct = 1.5x weight "
            "(anti-majority-class bias), normalized to 0–1."
        ),
        "action_schema": {
            "direction": {"type": "string", "enum": ["Bullish", "Bearish", "Neutral"]},
            "conviction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
    {
        "id": "long_term_conviction",
        "name": "Long-term Risk-Constrained Trading (Hard)",
        "description": (
            "Manage a position over 20 consecutive trading days (four weeks). "
            "Episode terminates early if portfolio drawdown exceeds 5% — the agent "
            "must avoid catastrophic losses, not just maximize return. "
            "Macro context (NIFTY50 trend, market regime) is included in each observation "
            "so the agent can condition decisions on market-wide conditions. "
            "Correct predictions with high conviction (≥0.7) score highest; "
            "overconfident wrong predictions are penalized. "
            "Ground-truth label: 60-day forward return threshold ±5.0%."
        ),
        "difficulty": "hard",
        "term": "long",
        "episode_steps": TASK_MAX_STEPS["long"],
        "grader_note": (
            "Score = conviction-calibrated accuracy: "
            "correct+conviction≥0.7 = 1.0, correct+conviction<0.7 = 0.5, "
            "wrong = 0.0, overconfident+wrong = −0.1. Normalized to 0–1."
        ),
        "action_schema": {
            "direction": {"type": "string", "enum": ["Bullish", "Bearish", "Neutral"]},
            "conviction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
]

TASK_BY_ID     = {t["id"]: t for t in TASKS}
TERM_TO_TASK_ID = {t["term"]: t["id"] for t in TASKS}

# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class IndicatorsAction(BaseModel):
    direction: str  = Field(..., description="Bullish=Long | Bearish=Short | Neutral=Flat")
    conviction: float = Field(0.5, ge=0.0, le=1.0, description="Confidence 0–1")


class IndicatorsObservation(BaseModel):
    symbol:        str
    date:          str
    term:          str
    current_price: float
    indicators:    Dict[str, Any]
    # Portfolio state — updated each step so actions causally affect next observation
    position:          int   = Field(0,   description="Current position: -1=Short, 0=Flat, 1=Long")
    unrealized_pnl:    float = Field(0.0, description="Total episode P&L as fraction (0.02 = +2%)")
    capital_remaining: float = Field(1.0, description="Capital remaining (1.0=start, 0.95=5% drawdown)")
    # Macro context — Task 3 (long-term) only
    macro: Optional[Dict[str, Any]] = Field(
        None,
        description="Macro context: nifty_trend, nifty_return_20d, market_regime (Task 3 only)"
    )


class StepResult(BaseModel):
    observation: Optional[IndicatorsObservation]
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetResult(BaseModel):
    observation: IndicatorsObservation
    info: Dict[str, Any]


class StateResult(BaseModel):
    session_id:           str
    current_observation:  Optional[IndicatorsObservation]
    episodes_completed:   int
    current_task:         Optional[str]


class GraderRequest(BaseModel):
    task_id: str
    episode_results: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "List of step dicts, each with keys: "
            "'ground_truth' (str), 'predicted' (str), 'conviction' (float)"
        ),
    )


class GraderResult(BaseModel):
    task_id:      str
    score:        float = Field(..., gt=0.0, lt=1.0)
    num_episodes: int
    breakdown:    Dict[str, Any]


# ─── Grader Logic ─────────────────────────────────────────────────────────────

def _clamp_score(score: float) -> float:
    """Clamp to strictly open interval (0, 1) as required by the validator."""
    return round(max(0.001, min(0.999, score)), 4)


def grade_task(task_id: str, episode_results: List[Dict[str, Any]]) -> GraderResult:
    """Deterministic grader for all 3 tasks. Returns a score in (0, 1) exclusive."""
    if not episode_results:
        return GraderResult(task_id=task_id, score=0.001, num_episodes=0, breakdown={})

    n = len(episode_results)

    if task_id == "short_term_direction":
        # Easy: simple directional accuracy
        correct = sum(
            1 for r in episode_results
            if r.get("predicted", "").capitalize() == r.get("ground_truth", "").capitalize()
        )
        score = _clamp_score(correct / n)
        breakdown = {"correct": correct, "total": n, "metric": "accuracy"}

    elif task_id == "medium_term_direction":
        # Medium: weighted accuracy (minority classes worth more)
        weighted_score = 0.0
        weighted_total = 0.0
        for r in episode_results:
            gt   = r.get("ground_truth", "").capitalize()
            pred = r.get("predicted", "").capitalize()
            w = 1.5 if gt in ("Bearish", "Neutral") else 1.0
            weighted_total += w
            if pred == gt:
                weighted_score += w
        score = _clamp_score(
            min(1.0, weighted_score / weighted_total) if weighted_total > 0 else 0.0
        )
        breakdown = {
            "weighted_score": round(weighted_score, 3),
            "weighted_total": round(weighted_total, 3),
            "metric": "weighted_accuracy",
        }

    elif task_id == "long_term_conviction":
        # Hard: direction + conviction calibration
        per_step_scores = []
        for r in episode_results:
            gt         = r.get("ground_truth", "").capitalize()
            pred       = r.get("predicted", "").capitalize()
            conviction = float(r.get("conviction", 0.5))
            correct    = pred == gt
            if correct and conviction >= 0.7:
                per_step_scores.append(1.0)
            elif correct and conviction < 0.7:
                per_step_scores.append(0.5)
            elif not correct and conviction >= 0.8:
                per_step_scores.append(-0.1)   # overconfident wrong
            else:
                per_step_scores.append(0.0)
        raw   = sum(per_step_scores) / n
        # Normalize [-0.1, 1.0] → (0, 1) exclusive
        score = _clamp_score((raw + 0.1) / 1.1)
        breakdown = {
            "per_step_scores": per_step_scores,
            "raw_mean":        round(raw, 4),
            "metric":          "conviction_calibrated_accuracy",
        }

    else:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

    return GraderResult(
        task_id=task_id,
        score=round(score, 4),
        num_episodes=n,
        breakdown=breakdown,
    )


# ─── Session state ────────────────────────────────────────────────────────────

class EnvSession:
    """
    Manages a single agent session across multi-step episodes.

    Portfolio tracking:
      position          : -1 (short), 0 (flat), 1 (long)
      capital           : starts at 1.0; updated each step by actual market return × position
      peak_capital      : running maximum capital (for drawdown calculation)
      unrealized_pnl    : capital - 1.0 (total episode P&L fraction)

    Reward formula (per step):
      pnl    = actual_1day_return × new_position − TRANSACTION_COST (if position changed)
      reward = clamp(pnl × REWARD_SCALE, −1.0, 1.1)

    Task 3 early termination:
      If drawdown = (peak_capital − capital) / peak_capital > DRAWDOWN_LIMIT,
      done=True before MAX_STEPS is reached.
    """

    def __init__(self, session_id: str, term: str = "medium"):
        self.session_id = session_id
        self.term       = term.lower()
        self.MAX_STEPS  = TASK_MAX_STEPS.get(self.term, 5)

        # Episode data: list of (raw_obs_dict, gt_label, actual_1day_return)
        self.episode_data:      List[Tuple[Dict[str, Any], str, float]] = []
        self.current_step:      int = 0
        self.current_obs:       Optional[IndicatorsObservation] = None
        self.current_gt:        Optional[str] = None
        self.episodes_completed: int = 0
        self.episode_history:   List[Dict[str, Any]] = []
        self.scenario_pool:     List[Dict[str, str]] = []

        # Portfolio state — reset each episode
        self.position:       int   = 0
        self.capital:        float = 1.0
        self.peak_capital:   float = 1.0
        self.unrealized_pnl: float = 0.0

    # ── helpers ──────────────────────────────────────────────────────────────

    def _reset_portfolio(self) -> None:
        self.position       = 0
        self.capital        = 1.0
        self.peak_capital   = 1.0
        self.unrealized_pnl = 0.0

    def _build_obs(self, raw_obs_dict: Dict[str, Any]) -> IndicatorsObservation:
        """Construct IndicatorsObservation injecting current portfolio state."""
        return IndicatorsObservation(
            **raw_obs_dict,
            position          = self.position,
            unrealized_pnl    = round(self.capital - 1.0, 6),
            capital_remaining = round(self.capital, 6),
        )

    def _get_scenario(self) -> Optional[Dict[str, str]]:
        if not self.scenario_pool:
            self.scenario_pool = generate_scenario_pool(
                symbols       = random.sample(NSE_UNIVERSE, min(20, len(NSE_UNIVERSE))),
                start_date    = "2020-01-01",
                end_date      = "2024-06-30",
                term          = self.term,
                max_scenarios = 2000,
            )
            random.shuffle(self.scenario_pool)
        return self.scenario_pool.pop() if self.scenario_pool else None

    # ── core interface ────────────────────────────────────────────────────────

    def reset(self) -> Optional[ResetResult]:
        """
        Start a new episode. Single yfinance call fetches the full episode window.
        Portfolio state is reset to cash (position=0, capital=1.0).
        """
        self._reset_portfolio()
        for _ in range(10):
            sc = self._get_scenario()
            if sc is None:
                return None
            steps = build_multi_step_episode(
                symbol        = sc["symbol"],
                start_date    = sc["date"],
                n_steps       = self.MAX_STEPS,
                term          = self.term,
                include_macro = (self.term == "long"),   # macro context for Task 3
            )
            if steps is not None:
                self.episode_data  = steps              # [(obs_dict, gt, actual_return), ...]
                self.current_step  = 0
                self.current_gt    = steps[0][1]
                self.current_obs   = self._build_obs(steps[0][0])
                return ResetResult(
                    observation = self.current_obs,
                    info = {
                        "session_id": self.session_id,
                        "term":       self.term,
                        "task_id":    TERM_TO_TASK_ID.get(self.term),
                        "step":       0,
                        "max_steps":  self.MAX_STEPS,
                    },
                )
        return None

    def step(self, action: IndicatorsAction) -> StepResult:
        if self.current_obs is None or self.current_gt is None:
            return StepResult(
                observation=None, reward=0.0, done=True,
                info={"error": "Call reset() first"},
            )

        direction = action.direction.strip().capitalize()
        if direction not in ("Bullish", "Bearish", "Neutral"):
            direction = "Neutral"

        direction_to_pos = {"Bullish": 1, "Neutral": 0, "Bearish": -1}
        new_position  = direction_to_pos[direction]
        prev_position = self.position

        # Transaction cost when changing position (realistic NSE brokerage + STT)
        cost = TRANSACTION_COST if new_position != prev_position else 0.0

        # Actual 1-day return from data — the market gives the reward
        _, _, actual_return = self.episode_data[self.current_step]

        # Portfolio P&L for this step
        pnl = actual_return * new_position - cost

        # Update capital
        self.capital      = max(0.0, self.capital * (1.0 + pnl))
        self.peak_capital = max(self.peak_capital, self.capital)
        drawdown = (
            (self.peak_capital - self.capital) / self.peak_capital
            if self.peak_capital > 0 else 0.0
        )

        # Update position
        self.position       = new_position
        self.unrealized_pnl = round(self.capital - 1.0, 6)

        # Scale reward: ×50 so a 2% daily return ≈ 1.0
        reward  = round(max(-1.0, min(1.1, pnl * REWARD_SCALE)), 4)
        correct = direction == self.current_gt

        self.episode_history.append({
            "ground_truth":      self.current_gt,
            "predicted":         direction,
            "conviction":        action.conviction,
            "reward":            reward,
            "actual_return_pct": round(actual_return * 100, 4),
            "position":          new_position,
            "capital":           round(self.capital, 6),
            "drawdown":          round(drawdown, 4),
        })

        self.current_step += 1
        # Task 3: terminate early on drawdown breach
        early_stop = (self.term == "long") and (drawdown > DRAWDOWN_LIMIT)
        done       = (self.current_step >= self.MAX_STEPS) or early_stop

        info = {
            "ground_truth":      self.current_gt,
            "predicted":         direction,
            "correct":           correct,
            "conviction":        action.conviction,
            "session_id":        self.session_id,
            "task_id":           TERM_TO_TASK_ID.get(self.term),
            "step":              self.current_step,
            "max_steps":         self.MAX_STEPS,
            "actual_return_pct": round(actual_return * 100, 4),
            "capital":           round(self.capital, 6),
            "drawdown":          round(drawdown, 4),
            "early_stop":        early_stop,
        }

        if not done:
            self.current_gt  = self.episode_data[self.current_step][1]
            self.current_obs = self._build_obs(self.episode_data[self.current_step][0])
        else:
            self.episodes_completed += 1
            self.current_obs = None
            self.current_gt  = None

        return StepResult(
            observation = self.current_obs,
            reward      = reward,
            done        = done,
            info        = info,
        )

    def state(self) -> StateResult:
        return StateResult(
            session_id          = self.session_id,
            current_observation = self.current_obs,
            episodes_completed  = self.episodes_completed,
            current_task        = TERM_TO_TASK_ID.get(self.term),
        )


# ─── Session store ────────────────────────────────────────────────────────────

_sessions: Dict[str, EnvSession] = {}

def _get_or_create(session_id: Optional[str] = None, term: str = "medium") -> EnvSession:
    if session_id is None:
        session_id = str(uuid.uuid4())
    if session_id not in _sessions:
        _sessions[session_id] = EnvSession(session_id, term=term)
    return _sessions[session_id]


# ─── HTTP endpoints (OpenEnv standard) ───────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "env":     "IndicatorsEnv",
        "version": "3.0.0",
        "tasks":   len(TASKS),
        "episode_steps": TASK_MAX_STEPS,
    }


@app.get("/tasks")
def get_tasks():
    """Returns all task definitions, action schemas, and episode lengths."""
    return {
        "tasks":  TASKS,
        "total":  len(TASKS),
        "action_schema": {
            "direction": {"type": "string", "enum": ["Bullish", "Bearish", "Neutral"]},
            "conviction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "episode_steps": TASK_MAX_STEPS,
    }


@app.post("/reset", response_model=ResetResult)
def reset(session_id: Optional[str] = None, term: str = "medium"):
    sess   = _get_or_create(session_id, term=term)
    result = sess.reset()
    if result is None:
        raise HTTPException(status_code=503, detail="Could not fetch data. Retry.")
    return result


@app.post("/step", response_model=StepResult)
def step(action: IndicatorsAction, session_id: Optional[str] = None):
    sess = _get_or_create(session_id)
    return sess.step(action)


@app.get("/state", response_model=StateResult)
def state(session_id: Optional[str] = None):
    sess = _get_or_create(session_id)
    return sess.state()


@app.post("/grader", response_model=GraderResult)
def grader(request: GraderRequest):
    """
    Score a completed episode set against a specific task grader.
    Provide episode_results as list of {ground_truth, predicted, conviction}.
    Returns a score in (0.0, 1.0).
    """
    if request.task_id not in TASK_BY_ID:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id: {request.task_id}. Valid: {list(TASK_BY_ID.keys())}",
        )
    return grade_task(request.task_id, request.episode_results)


@app.get("/baseline")
def baseline():
    """
    Run the built-in random baseline agent across all 3 tasks (10 episodes each).
    Returns reproducible baseline scores. Expected overall_mean ≈ 0.33.
    Episode steps: short=5×10=50, medium=10×10=100, long=20×10=200.
    """
    results = {}
    random.seed(42)

    for task in TASKS:
        term    = task["term"]
        task_id = task["id"]
        sess    = EnvSession(session_id=f"baseline-{task_id}", term=term)
        episode_results: List[Dict[str, Any]] = []

        for _ in range(10):
            reset_result = sess.reset()
            if reset_result is None:
                continue
            done = False
            while not done:
                action = IndicatorsAction(
                    direction  = random.choice(["Bullish", "Bearish", "Neutral"]),
                    conviction = round(random.uniform(0.3, 0.9), 2),
                )
                step_result = sess.step(action)
                episode_results.append({
                    "ground_truth": step_result.info.get("ground_truth", ""),
                    "predicted":    action.direction,
                    "conviction":   action.conviction,
                })
                done = step_result.done

        grader_result = grade_task(task_id, episode_results)
        results[task_id] = {
            "task_name":    task["name"],
            "difficulty":   task["difficulty"],
            "score":        grader_result.score,
            "num_episodes": grader_result.num_episodes,
            "breakdown":    grader_result.breakdown,
        }

    return {
        "agent":         "random_baseline",
        "seed":          42,
        "tasks":         results,
        "overall_mean":  round(sum(r["score"] for r in results.values()) / len(results), 4),
    }


# ─── WebSocket endpoint (OpenEnv /ws) ────────────────────────────────────────

@app.websocket("/ws")
async def websocket_env(ws: WebSocket, session_id: Optional[str] = None, term: str = "medium"):
    await ws.accept()
    sess = _get_or_create(session_id, term=term)
    logger.info(f"[WS] Session {sess.session_id} connected (term={term})")
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            method = msg.get("method", "")

            if method == "reset":
                result = sess.reset()
                if result is None:
                    await ws.send_json({"error": "No data available"})
                else:
                    await ws.send_text(result.model_dump_json())

            elif method == "step":
                action = IndicatorsAction(**msg.get("action", {}))
                result = sess.step(action)
                await ws.send_text(result.model_dump_json())

            elif method == "state":
                result = sess.state()
                await ws.send_text(result.model_dump_json())

            else:
                await ws.send_json({"error": f"Unknown method: {method}"})

    except WebSocketDisconnect:
        logger.info(f"[WS] Session {sess.session_id} disconnected")
    except Exception as e:
        logger.error(f"[WS] Error: {e}")
        await ws.close()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # workers=1 required: _sessions is an in-memory dict, not shared across processes
    uvicorn.run("indicators_env:app", host="0.0.0.0", port=7860, reload=False, workers=1)
