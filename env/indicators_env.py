"""
indicators_env.py — OpenEnv-compatible RL environment for technical indicators prediction.

Implements the full OpenEnv standard interface:
  - HTTP endpoints: reset / step / state / tasks / grader / baseline / health
  - WebSocket (/ws) for low-latency sequential step() calls during RL training

Env spec:
  Observation : JSON dict of full indicator snapshot + [TERM: MEDIUM] token
  Action      : {"direction": "Bullish"|"Bearish"|"Neutral", "conviction": float 0-1}
  Reward      : Shaped 0.0–1.1 based on correctness + conviction calibration

Tasks:
  Task 1 (Easy)   : SHORT  term (5d,  ±1.0% threshold)
  Task 2 (Medium) : MEDIUM term (20d, ±2.5% threshold)  ← default
  Task 3 (Hard)   : LONG   term (60d, ±5.0% threshold) + conviction >= 0.7 required
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from typing import Any, Dict, List, Optional

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
        "OpenEnv-compatible RL environment for NSE technical indicators prediction. "
        "An AI agent receives a rich indicator snapshot and must predict the directional "
        "price move over a configurable forward window (short/medium/long term). "
        "Built for the Meta × PyTorch Hackathon."
    ),
    version="2.0.0",
)

# ─── Task Definitions ─────────────────────────────────────────────────────────

TASKS = [
    {
        "id": "short_term_direction",
        "name": "Short-term Direction (Easy)",
        "description": (
            "Predict 5-day forward price direction from technical indicators. "
            "Threshold: ±1.0%. More Neutral outcomes expected due to market noise."
        ),
        "difficulty": "easy",
        "term": "short",
        "grader_note": "Score = fraction of correct directional predictions over 10 episodes.",
        "action_schema": {
            "direction": {"type": "string", "enum": ["Bullish", "Bearish", "Neutral"]},
            "conviction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
    {
        "id": "medium_term_direction",
        "name": "Medium-term Direction (Medium)",
        "description": (
            "Predict 20-day forward price direction from technical indicators. "
            "Threshold: ±2.5%. Balanced Bullish/Bearish/Neutral distribution."
        ),
        "difficulty": "medium",
        "term": "medium",
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
        "name": "Long-term Conviction (Hard)",
        "description": (
            "Predict 60-day forward price direction AND demonstrate calibrated conviction. "
            "Threshold: ±5.0%. Correct direction WITH conviction >= 0.7 scores 1.0. "
            "Correct direction with low conviction scores 0.5. Wrong + high conviction penalized."
        ),
        "difficulty": "hard",
        "term": "long",
        "grader_note": (
            "Score = mean of per-episode scores: 1.0 (correct + conviction>=0.7), "
            "0.5 (correct + conviction<0.7), 0.0 (wrong), -0.1 (wrong + conviction>=0.8). "
            "Normalized to 0–1 range."
        ),
        "action_schema": {
            "direction": {"type": "string", "enum": ["Bullish", "Bearish", "Neutral"]},
            "conviction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
]

TASK_BY_ID = {t["id"]: t for t in TASKS}
TERM_TO_TASK_ID = {t["term"]: t["id"] for t in TASKS}

# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class IndicatorsAction(BaseModel):
    direction: str = Field(..., description="Predicted direction: Bullish | Bearish | Neutral")
    conviction: float = Field(0.5, ge=0.0, le=1.0, description="Confidence in prediction (0-1)")


class IndicatorsObservation(BaseModel):
    symbol: str
    date: str
    term: str
    current_price: float
    indicators: Dict[str, Any]


class StepResult(BaseModel):
    observation: Optional[IndicatorsObservation]
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetResult(BaseModel):
    observation: IndicatorsObservation
    info: Dict[str, Any]


class StateResult(BaseModel):
    session_id: str
    current_observation: Optional[IndicatorsObservation]
    episodes_completed: int
    current_task: Optional[str]


class GraderRequest(BaseModel):
    task_id: str
    episode_results: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "List of episode dicts, each with keys: "
            "'ground_truth' (str), 'predicted' (str), 'conviction' (float)"
        ),
    )


class GraderResult(BaseModel):
    task_id: str
    score: float = Field(..., gt=0.0, lt=1.0)
    num_episodes: int
    breakdown: Dict[str, Any]


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
        # Easy: simple accuracy
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
            gt = r.get("ground_truth", "").capitalize()
            pred = r.get("predicted", "").capitalize()
            w = 1.5 if gt in ("Bearish", "Neutral") else 1.0
            weighted_total += w
            if pred == gt:
                weighted_score += w
        score = _clamp_score(min(1.0, weighted_score / weighted_total) if weighted_total > 0 else 0.0)
        breakdown = {"weighted_score": round(weighted_score, 3), "weighted_total": round(weighted_total, 3), "metric": "weighted_accuracy"}

    elif task_id == "long_term_conviction":
        # Hard: direction + conviction calibration
        per_episode_scores = []
        for r in episode_results:
            gt = r.get("ground_truth", "").capitalize()
            pred = r.get("predicted", "").capitalize()
            conviction = float(r.get("conviction", 0.5))
            correct = pred == gt
            if correct and conviction >= 0.7:
                per_episode_scores.append(1.0)
            elif correct and conviction < 0.7:
                per_episode_scores.append(0.5)
            elif not correct and conviction >= 0.8:
                per_episode_scores.append(-0.1)  # overconfident wrong
            else:
                per_episode_scores.append(0.0)
        raw = sum(per_episode_scores) / n
        # Normalize from [-0.1, 1.0] to (0, 1) exclusive
        score = _clamp_score((raw + 0.1) / 1.1)
        breakdown = {"per_episode_scores": per_episode_scores, "raw_mean": round(raw, 4), "metric": "conviction_calibrated_accuracy"}

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
    MAX_STEPS = 5  # One trading week per episode

    def __init__(self, session_id: str, term: str = "medium"):
        self.session_id = session_id
        self.term = term.lower()
        self.current_step: int = 0
        self.episode_data: List = []   # List of (IndicatorsObservation, gt_str)
        self.current_obs: Optional[IndicatorsObservation] = None
        self.current_gt: Optional[str] = None
        self.episodes_completed: int = 0
        self.episode_history: List[Dict[str, Any]] = []
        self.scenario_pool: List[Dict[str, str]] = []

    def _get_scenario(self) -> Optional[Dict[str, str]]:
        if not self.scenario_pool:
            self.scenario_pool = generate_scenario_pool(
                symbols=random.sample(NSE_UNIVERSE, min(20, len(NSE_UNIVERSE))),
                start_date="2020-01-01",
                end_date="2024-06-30",
                term=self.term,
                max_scenarios=2000,
            )
            random.shuffle(self.scenario_pool)
        return self.scenario_pool.pop() if self.scenario_pool else None

    def reset(self) -> Optional[ResetResult]:
        """Start a new 5-step episode. Single yfinance call fetches the full week."""
        for _ in range(10):
            sc = self._get_scenario()
            if sc is None:
                return None
            steps = build_multi_step_episode(
                symbol=sc["symbol"],
                start_date=sc["date"],
                n_steps=self.MAX_STEPS,
                term=self.term,
            )
            if steps is not None:
                self.episode_data = [
                    (IndicatorsObservation(**obs_dict), gt)
                    for obs_dict, gt in steps
                ]
                self.current_step = 0
                self.current_obs, self.current_gt = self.episode_data[0]
                return ResetResult(
                    observation=self.current_obs,
                    info={
                        "session_id": self.session_id,
                        "term": self.term,
                        "task_id": TERM_TO_TASK_ID.get(self.term),
                        "step": 0,
                        "max_steps": self.MAX_STEPS,
                    },
                )
        return None

    def step(self, action: IndicatorsAction) -> StepResult:
        if self.current_obs is None or self.current_gt is None:
            return StepResult(
                observation=None, reward=0.0, done=True,
                info={"error": "Call reset() first"}
            )

        correct = action.direction.strip().capitalize() == self.current_gt

        # Shaped reward: conviction calibration — range [-0.1, 1.1]
        reward = 1.0 if correct else 0.0
        if correct and action.conviction >= 0.6:
            reward = 1.0 + 0.1 * (action.conviction - 0.6)
        elif not correct and action.conviction >= 0.8:
            reward = -0.1
        reward = round(max(-0.1, min(1.1, reward)), 4)

        # Store for grader
        self.episode_history.append({
            "ground_truth": self.current_gt,
            "predicted": action.direction.strip().capitalize(),
            "conviction": action.conviction,
            "reward": reward,
        })

        self.current_step += 1
        done = self.current_step >= self.MAX_STEPS

        info = {
            "ground_truth": self.current_gt,
            "predicted": action.direction,
            "correct": correct,
            "conviction": action.conviction,
            "session_id": self.session_id,
            "task_id": TERM_TO_TASK_ID.get(self.term),
            "step": self.current_step,
            "max_steps": self.MAX_STEPS,
        }

        if not done:
            self.current_obs, self.current_gt = self.episode_data[self.current_step]
        else:
            self.episodes_completed += 1
            self.current_obs = None
            self.current_gt = None

        return StepResult(
            observation=self.current_obs,
            reward=reward,
            done=done,
            info=info,
        )

    def state(self) -> StateResult:
        return StateResult(
            session_id=self.session_id,
            current_observation=self.current_obs,
            episodes_completed=self.episodes_completed,
            current_task=TERM_TO_TASK_ID.get(self.term),
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
    return {"status": "ok", "env": "IndicatorsEnv", "version": "2.0.0", "tasks": len(TASKS)}


@app.get("/tasks")
def get_tasks():
    """Returns all task definitions and action schemas."""
    return {
        "tasks": TASKS,
        "total": len(TASKS),
        "action_schema": {
            "direction": {"type": "string", "enum": ["Bullish", "Bearish", "Neutral"]},
            "conviction": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    }


@app.post("/reset", response_model=ResetResult)
def reset(session_id: Optional[str] = None, term: str = "medium"):
    sess = _get_or_create(session_id, term=term)
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
    Returns a 0.0–1.0 score.
    """
    if request.task_id not in TASK_BY_ID:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {request.task_id}. Valid: {list(TASK_BY_ID.keys())}")
    return grade_task(request.task_id, request.episode_results)


@app.get("/baseline")
def baseline():
    """
    Trigger the built-in random baseline agent across all 3 tasks (10 episodes each).
    Returns reproducible baseline scores.
    """
    results = {}
    random.seed(42)  # Deterministic

    for task in TASKS:
        term = task["term"]
        task_id = task["id"]
        session_id = f"baseline-{task_id}"

        # Fresh session
        sess = EnvSession(session_id=session_id, term=term)
        episode_results = []

        for _ in range(10):
            reset_result = sess.reset()
            if reset_result is None:
                continue
            done = False
            while not done:
                action = IndicatorsAction(
                    direction=random.choice(["Bullish", "Bearish", "Neutral"]),
                    conviction=round(random.uniform(0.3, 0.9), 2),
                )
                step_result = sess.step(action)
                episode_results.append({
                    "ground_truth": step_result.info.get("ground_truth", ""),
                    "predicted": action.direction,
                    "conviction": action.conviction,
                })
                done = step_result.done

        grader_result = grade_task(task_id, episode_results)
        results[task_id] = {
            "task_name": task["name"],
            "difficulty": task["difficulty"],
            "score": grader_result.score,
            "num_episodes": grader_result.num_episodes,
            "breakdown": grader_result.breakdown,
        }

    return {
        "agent": "random_baseline",
        "seed": 42,
        "tasks": results,
        "overall_mean": round(sum(r["score"] for r in results.values()) / len(results), 4),
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
