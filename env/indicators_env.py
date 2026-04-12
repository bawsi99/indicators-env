"""
indicators_env.py — IndicatorsEnv v4.0: Multi-stock Relative Alpha MDP

OpenEnv-compatible RL environment for NSE equity analysis.

Full OpenEnv interface:
  HTTP  : GET  /health  /tasks
          POST /reset  /step  /grader
          GET  /state  /baseline
  WebSocket: /ws

Env design (v4.0) — Multi-stock Relative Alpha MDP:
  At each step the agent observes 3 stocks from the same NSE sector.
  It picks ONE stock (or passes with NONE) and declares a direction.

  Observation:
    - 3 stocks from the same sector (same stocks for the whole episode)
    - Full technical indicator snapshot per stock
    - RSI + price-momentum signal history (accumulated across steps)
    - Macro context in Task 3 (NIFTY50 trend, market regime)

  Action:
    {"stock": "HDFCBANK", "direction": "Bullish"|"Bearish"|"NONE", "conviction": 0.8}
    NONE = skip this step (preserve selective participation)

  Reward (market-neutral alpha):
    alpha  = chosen_stock_period_return − sector_avg_period_return
    reward = alpha × direction_sign × conviction × REWARD_SCALE
    → random policy earns ~0 expected reward (sector avg cancels market beta)
    → skilled policy earns consistently positive reward (correct stock selection)
    → Kelly conviction semantics: fraction of virtual wealth wagered per step

  Step spacing:
    short  :  1 trading day per step → GT = 1-day return   (5 steps  = 1 week)
    medium :  5 trading days per step → GT = 5-day return  (10 steps = 10 weeks)
    long   : 20 trading days per step → GT = 20-day return (15 steps = 15 months)
    GT window = step spacing → zero overlap, reward and GT measure identical period.

  Task 3 extras:
    - Drawdown limit: virtual capital terminates at 10% drawdown
    - Macro context (NIFTY50) injected into each observation

Built for the Meta × PyTorch Hackathon.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import uuid
from statistics import mean
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

try:
    import yf_patch
    yf_patch.patch_yfinance_globally()
except ImportError:
    pass

import pandas as pd
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from data_loader import (
    NSE_UNIVERSE, 
    SECTOR_GROUPS, 
    TERM_WINDOWS, 
    TERM_THRESHOLDS,
    STEP_SPACING,
    PERIOD_THRESHOLDS,
    build_multi_stock_episode,
    fetch_macro_context
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="IndicatorsEnv",
    description=(
        "IndicatorsEnv v4.0 — Multi-stock Relative Alpha MDP for NSE equity analysis. "
        "At each step the agent observes 3 stocks from the same NSE sector and picks one "
        "(or passes). Reward = (chosen stock return − sector average) × direction × conviction. "
        "Market-neutral: random policy earns ~0, skilled policy earns positive alpha. "
        "Three tasks of increasing difficulty. Built for the Meta × PyTorch Hackathon."
    ),
    version="4.0.0",
)

# ─── Environment constants ────────────────────────────────────────────────────

TASK_MAX_STEPS: Dict[str, int] = {
    "short":  5,    # 5 × 1-day  = 1 trading week
    "medium": 10,   # 10 × 5-day = 10 trading weeks
    "long":   15,   # 15 × 20-day = 15 months
}

REWARD_SCALE   = 50.0    # ×50: 2% alpha × conviction 0.7 × 50 ≈ 0.7 reward
DRAWDOWN_LIMIT = 0.10    # 10% virtual capital drawdown terminates Task 3 early

# ─── Task definitions ─────────────────────────────────────────────────────────

TASKS = [
    {
        "id": "short_term_direction",
        "name": "Short-term Relative Alpha (Easy)",
        "description": (
            "5 steps, 1 trading day apart (= 1 week of daily observations). "
            "At each step observe 3 stocks from the same NSE sector. "
            "Pick one stock and declare Bullish (long), Bearish (short), or NONE (pass). "
            "Reward = (chosen_return − sector_avg) × direction × conviction × 50. "
            "Market-neutral: sector beta cancels — the agent profits from within-sector "
            "relative momentum, not from broad market moves. "
            "GT = 1-day forward return (±0.3% threshold)."
        ),
        "difficulty": "easy",
        "term": "short",
        "episode_steps": TASK_MAX_STEPS["short"],
        "step_spacing_days": STEP_SPACING["short"],
        "grader_note": "Directional accuracy on active (non-NONE) steps.",
        "action_schema": {
            "stock":     {"type": "string", "description": "NSE symbol or 'NONE' to skip"},
            "direction": {"type": "string", "enum": ["Bullish", "Bearish", "NONE"]},
            "conviction":{"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
    {
        "id": "medium_term_direction",
        "name": "Medium-term Relative Alpha (Medium)",
        "description": (
            "10 steps, 5 trading days apart (= 10 weeks, one observation per week). "
            "Same 3-stock multi-stock structure. "
            "Signal history accumulates across steps — RSI trends and price momentum "
            "are visible for all 3 stocks at each step, enabling sequential inference. "
            "Reward = (chosen_return − sector_avg) × direction × conviction × 50. "
            "GT = 5-day forward return (±1.5% threshold). "
            "Grader gives extra weight to Bearish/Neutral correct calls (anti-majority-bias)."
        ),
        "difficulty": "medium",
        "term": "medium",
        "episode_steps": TASK_MAX_STEPS["medium"],
        "step_spacing_days": STEP_SPACING["medium"],
        "grader_note": (
            "Weighted accuracy on active steps: "
            "Bearish/Neutral correct = 1.5× weight. "
            "Participation bonus: score × (0.9 + 0.1 × active_rate)."
        ),
        "action_schema": {
            "stock":     {"type": "string", "description": "NSE symbol or 'NONE' to skip"},
            "direction": {"type": "string", "enum": ["Bullish", "Bearish", "NONE"]},
            "conviction":{"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
    {
        "id": "long_term_conviction",
        "name": "Long-term Risk-Constrained Alpha (Hard)",
        "description": (
            "15 steps, 20 trading days apart (= 15 months, one observation per month). "
            "Spans multiple market regimes — signal history essential for detecting "
            "multi-month relative momentum shifts within the sector. "
            "Macro context (NIFTY50 trend, market regime) added to each observation. "
            "Episode terminates early if virtual capital drawdown exceeds 10% — "
            "the agent must manage risk as well as generate alpha. "
            "Correct calls with conviction ≥ 0.7 score highest; "
            "overconfident wrong predictions penalized (conviction ≥ 0.8 on wrong). "
            "GT = 20-day forward return (±2.5% threshold)."
        ),
        "difficulty": "hard",
        "term": "long",
        "episode_steps": TASK_MAX_STEPS["long"],
        "step_spacing_days": STEP_SPACING["long"],
        "grader_note": (
            "Conviction-calibrated accuracy on active steps: "
            "correct+conviction≥0.7 = 1.0, correct+conviction<0.7 = 0.5, "
            "wrong = 0.0, overconfident wrong (conviction≥0.8) = −0.1. "
            "Normalized to (0, 1)."
        ),
        "action_schema": {
            "stock":     {"type": "string", "description": "NSE symbol or 'NONE' to skip"},
            "direction": {"type": "string", "enum": ["Bullish", "Bearish", "NONE"]},
            "conviction":{"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
]

TASK_BY_ID      = {t["id"]: t for t in TASKS}
TERM_TO_TASK_ID = {t["term"]: t["id"] for t in TASKS}

# ─── Pydantic schemas ─────────────────────────────────────────────────────────


class StockState(BaseModel):
    """Per-stock snapshot at one episode step."""
    symbol:             str
    current_price:      float
    rsi_14:             float
    rsi_trend:          str          # "up" | "down" | "flat"
    price_momentum_pct: float        # cumulative return vs episode start (%)
    indicators:         Dict[str, Any]


class MultiStockObservation(BaseModel):
    """Full observation returned by reset() and step()."""
    step:             int               # 1-indexed current step
    max_steps:        int
    term:             str               # SHORT | MEDIUM | LONG
    sector:           str               # e.g. "banking"
    available_stocks: List[str]         # 3 symbols
    stocks:           Dict[str, StockState]   # symbol → full state
    signal_history:   List[Dict[str, Any]]    # chronological pick log
    macro:            Optional[Dict[str, Any]] = None   # Task 3 only


class MultiStockAction(BaseModel):
    stock:     str   = Field(..., description="NSE symbol to pick, or 'NONE' to skip")
    direction: str   = Field(..., description="Bullish | Bearish | NONE")
    conviction: float = Field(0.5, ge=0.0, le=1.0, description="Confidence 0–1")


class StepResult(BaseModel):
    observation: Optional[MultiStockObservation]
    reward:      float
    done:        bool
    info:        Dict[str, Any]


class ResetResult(BaseModel):
    observation: MultiStockObservation
    info:        Dict[str, Any]


class StateResult(BaseModel):
    session_id:          str
    current_observation: Optional[MultiStockObservation]
    episodes_completed:  int
    current_task:        Optional[str]


class GraderRequest(BaseModel):
    task_id: str
    episode_results: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "List of step dicts, each with keys: "
            "'predicted' (str: Bullish|Bearish|NONE), "
            "'ground_truth' (str: Bullish|Bearish|Neutral|N/A), "
            "'conviction' (float)"
        ),
    )


class GraderResult(BaseModel):
    task_id:      str
    score:        float = Field(..., gt=0.0, lt=1.0)
    num_episodes: int
    breakdown:    Dict[str, Any]


# ─── Grader logic ─────────────────────────────────────────────────────────────


def _clamp_score(score: float) -> float:
    """Clamp to strictly open interval (0, 1) as required by the OpenEnv validator."""
    return round(max(0.001, min(0.999, score)), 4)


def grade_task(task_id: str, episode_results: List[Dict[str, Any]]) -> GraderResult:
    """
    Score episode results for a given task.

    Active steps = steps where predicted != 'NONE'.
    Grader evaluates only active steps; passing silently does not harm the score
    but forfeits the opportunity to score points.
    """
    if not episode_results:
        return GraderResult(
            task_id=task_id, score=0.001, num_episodes=0,
            breakdown={"note": "No results submitted"}
        )

    n_total  = len(episode_results)
    active   = [
        r for r in episode_results
        if str(r.get("predicted", "NONE")).capitalize() != "None"
        and str(r.get("predicted", "NONE")).upper() != "NONE"
    ]
    n_active = len(active)

    if n_active == 0:
        return GraderResult(
            task_id=task_id, score=0.001, num_episodes=n_total,
            breakdown={"note": "All steps passed (NONE)", "active_steps": 0}
        )

    participation_rate = n_active / n_total

    if task_id == "short_term_direction":
        correct = sum(
            1 for r in active
            if str(r.get("predicted","")).capitalize() == str(r.get("ground_truth","")).capitalize()
        )
        raw_score = correct / n_active
        score = _clamp_score(raw_score)
        breakdown = {
            "correct": correct,
            "active_steps": n_active,
            "total_steps": n_total,
            "participation_rate": round(participation_rate, 3),
            "metric": "directional_accuracy",
        }

    elif task_id == "medium_term_direction":
        weighted_score = 0.0
        weighted_total = 0.0
        for r in active:
            gt   = str(r.get("ground_truth", "")).capitalize()
            pred = str(r.get("predicted", "")).capitalize()
            w = 1.5 if gt in ("Bearish", "Neutral") else 1.0
            weighted_total += w
            if pred == gt:
                weighted_score += w
        base = (weighted_score / weighted_total) if weighted_total > 0 else 0.0
        # Participation bonus: agents that selectively engage vs. pass every step
        adjusted = base * (0.9 + 0.1 * participation_rate)
        score = _clamp_score(adjusted)
        breakdown = {
            "weighted_score": round(weighted_score, 3),
            "weighted_total": round(weighted_total, 3),
            "participation_rate": round(participation_rate, 3),
            "active_steps": n_active,
            "total_steps": n_total,
            "metric": "weighted_accuracy_with_participation",
        }

    elif task_id == "long_term_conviction":
        per_step: List[float] = []
        for r in active:
            gt         = str(r.get("ground_truth", "")).capitalize()
            pred       = str(r.get("predicted",    "")).capitalize()
            conviction = float(r.get("conviction", 0.5))
            correct    = (pred == gt)
            if correct and conviction >= 0.7:
                per_step.append(1.0)
            elif correct and conviction < 0.7:
                per_step.append(0.5)
            elif not correct and conviction >= 0.8:
                per_step.append(-0.1)   # overconfident wrong
            else:
                per_step.append(0.0)
        raw = sum(per_step) / n_active
        # Normalize [-0.1, 1.0] → (0, 1) exclusive
        score = _clamp_score((raw + 0.1) / 1.1)
        breakdown = {
            "per_step_scores": per_step,
            "raw_mean": round(raw, 4),
            "active_steps": n_active,
            "total_steps": n_total,
            "participation_rate": round(participation_rate, 3),
            "metric": "conviction_calibrated_accuracy",
        }

    else:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

    return GraderResult(
        task_id=task_id,
        score=round(score, 4),
        num_episodes=n_total,
        breakdown=breakdown,
    )


# ─── Session state ────────────────────────────────────────────────────────────


class EnvSession:
    """
    Manages a single agent session across multi-stock episodes.

    State per episode:
      episode_steps    : list of n_steps dicts from build_multi_stock_episode
      current_step_idx : 0-indexed pointer into episode_steps
      signal_history   : accumulated RSI/momentum log shown in each observation
      virtual_capital  : starts 1.0; updated by scaled alpha reward (Task 3 drawdown)

    Reward (per step):
      alpha  = chosen_stock_period_return − mean(all_3_stocks_period_return)
      reward = alpha × direction_sign × conviction × REWARD_SCALE
      NONE action → reward = 0.0
    """

    def __init__(self, session_id: str, term: str = "medium"):
        self.session_id     = session_id
        self.term           = term.lower()
        self.MAX_STEPS      = TASK_MAX_STEPS.get(self.term, 5)

        # Episode data: list of step dicts from build_multi_stock_episode
        self.episode_steps:     List[Dict[str, Any]] = []
        self.current_step_idx:  int = 0
        self.current_obs:       Optional[MultiStockObservation] = None
        self.sector:            str = ""
        self.symbols:           List[str] = []

        self.episodes_completed: int = 0
        self.episode_history:    List[Dict[str, Any]] = []
        self.signal_history:     List[Dict[str, Any]] = []

        # Task 3: virtual capital for drawdown tracking
        self.virtual_capital: float = 1.0
        self.peak_capital:    float = 1.0

        # Lazy scenario pool
        self.scenario_pool: List[Dict[str, Any]] = []

    # ── helpers ──────────────────────────────────────────────────────────────

    def _reset_episode_state(self) -> None:
        self.episode_steps    = []
        self.current_step_idx = 0
        self.signal_history   = []
        self.virtual_capital  = 1.0
        self.peak_capital     = 1.0

    def _get_scenario(self) -> Optional[Dict[str, Any]]:
        """Pop a (sector, symbols, date) scenario from the pool, refilling if needed."""
        if not self.scenario_pool:
            sectors   = list(SECTOR_GROUPS.keys())
            # Date range safely within available yfinance data
            # For long term: leave 15×20=300 days of future data → cap at 2022-12-31
            end_date = {
                "short":  "2024-06-30",
                "medium": "2023-12-31",
                "long":   "2022-06-30",
            }.get(self.term, "2023-12-31")

            dates = (
                pd.bdate_range("2020-01-01", end_date, freq="15B")
                .strftime("%Y-%m-%d")
                .tolist()
            )
            for sector in sectors:
                syms = SECTOR_GROUPS[sector]
                sampled_dates = random.sample(dates, min(60, len(dates)))
                for date in sampled_dates:
                    self.scenario_pool.append({
                        "sector":  sector,
                        "symbols": random.sample(syms, min(3, len(syms))),
                        "date":    date,
                    })
            random.shuffle(self.scenario_pool)

        return self.scenario_pool.pop() if self.scenario_pool else None

    def _build_obs(self, step_idx: int) -> MultiStockObservation:
        """Construct MultiStockObservation for the current step."""
        raw = self.episode_steps[step_idx]
        stocks_dict: Dict[str, StockState] = {}
        for s in raw["stocks"]:
            od = s["obs_dict"]
            stocks_dict[s["symbol"]] = StockState(
                symbol             = od["symbol"],
                current_price      = od["current_price"],
                rsi_14             = od["rsi_14"],
                rsi_trend          = od["rsi_trend"],
                price_momentum_pct = od["price_momentum_pct"],
                indicators         = od["indicators"],
            )

        return MultiStockObservation(
            step             = step_idx + 1,          # 1-indexed
            max_steps        = self.MAX_STEPS,
            term             = self.term.upper(),
            sector           = self.sector,
            available_stocks = self.symbols,
            stocks           = stocks_dict,
            signal_history   = list(self.signal_history),
            macro            = raw.get("macro"),
        )

    # ── core interface ────────────────────────────────────────────────────────

    def reset(self) -> Optional[ResetResult]:
        """
        Start a new episode. Picks a sector + 3 stocks, fetches data in 3 yfinance
        calls (one per stock), builds the full episode in one pass.
        """
        self._reset_episode_state()

        for _ in range(10):
            sc = self._get_scenario()
            if sc is None:
                return None

            steps = build_multi_stock_episode(
                symbols      = sc["symbols"],
                start_date   = sc["date"],
                n_steps      = self.MAX_STEPS,
                term         = self.term,
                include_macro= (self.term == "long"),
            )
            if steps is not None:
                self.episode_steps = steps
                self.sector        = sc["sector"]
                self.symbols       = sc["symbols"]
                self.current_obs   = self._build_obs(0)
                return ResetResult(
                    observation = self.current_obs,
                    info = {
                        "session_id": self.session_id,
                        "term":       self.term,
                        "task_id":    TERM_TO_TASK_ID.get(self.term),
                        "sector":     self.sector,
                        "symbols":    self.symbols,
                        "step":       0,
                        "max_steps":  self.MAX_STEPS,
                    },
                )
        return None

    def step(self, action: MultiStockAction) -> StepResult:
        if not self.episode_steps or self.current_step_idx >= len(self.episode_steps):
            return StepResult(
                observation=None, reward=0.0, done=True,
                info={"error": "Call reset() first"},
            )

        raw         = self.episode_steps[self.current_step_idx]
        stock_rows  = {s["symbol"]: s for s in raw["stocks"]}

        # ── Parse action ──────────────────────────────────────────────────────
        stock     = action.stock.strip().upper()
        direction = action.direction.strip().capitalize()
        if direction not in ("Bullish", "Bearish"):
            direction = "NONE"
        if stock == "NONE" or direction == "NONE":
            stock, direction = "NONE", "NONE"
        elif stock not in stock_rows:
            # Invalid symbol → treat as NONE
            stock, direction = "NONE", "NONE"

        conviction = action.conviction

        # ── Reward: market-neutral alpha ──────────────────────────────────────
        all_returns = [s["actual_period_return"] for s in raw["stocks"]]
        sector_avg  = mean(all_returns) if all_returns else 0.0

        if direction == "NONE":
            reward        = 0.0
            alpha         = 0.0
            chosen_return = 0.0
            chosen_gt     = "N/A"
        else:
            chosen_return = stock_rows[stock]["actual_period_return"]
            alpha         = chosen_return - sector_avg
            direction_sign = 1 if direction == "Bullish" else -1
            raw_reward     = alpha * direction_sign * conviction * REWARD_SCALE
            reward         = round(max(-1.5, min(1.5, raw_reward)), 4)
            chosen_gt      = stock_rows[stock]["gt"]

            # Task 3: update virtual capital for drawdown tracking
            if self.term == "long":
                pnl_fraction = alpha * direction_sign * conviction
                self.virtual_capital  = max(0.0, self.virtual_capital + pnl_fraction)
                self.peak_capital     = max(self.peak_capital, self.virtual_capital)

        correct   = (direction != "NONE") and (direction == chosen_gt)
        drawdown  = (
            (self.peak_capital - self.virtual_capital) / self.peak_capital
            if self.peak_capital > 0 else 0.0
        )

        # ── Update signal history ─────────────────────────────────────────────
        self.signal_history.append({
            "step":              self.current_step_idx + 1,
            "picked_stock":      stock,
            "direction":         direction,
            "conviction":        conviction,
            "ground_truth":      chosen_gt,
            "correct":           correct if direction != "NONE" else None,
            "alpha_pct":         round(alpha * 100, 3),
            "reward":            reward,
            # RSI snapshot of all stocks at this step
            "rsi_snapshot": {
                s["symbol"]: round(s["obs_dict"]["rsi_14"], 1)
                for s in raw["stocks"]
            },
        })

        self.episode_history.append({
            "step":         self.current_step_idx + 1,
            "stock":        stock,
            "direction":    direction,
            "ground_truth": chosen_gt,
            "correct":      correct,
            "conviction":   conviction,
            "reward":       reward,
            "alpha_pct":    round(alpha * 100, 3),
            "chosen_return_pct": round(chosen_return * 100, 4),
            "sector_avg_pct":    round(sector_avg * 100, 4),
        })

        # ── Advance step ──────────────────────────────────────────────────────
        self.current_step_idx += 1

        early_stop = (self.term == "long") and (drawdown > DRAWDOWN_LIMIT)
        done       = (self.current_step_idx >= self.MAX_STEPS) or early_stop

        info = {
            "step":              self.current_step_idx,
            "max_steps":         self.MAX_STEPS,
            "session_id":        self.session_id,
            "task_id":           TERM_TO_TASK_ID.get(self.term),
            "sector":            self.sector,
            "chosen_stock":      stock,
            "direction":         direction,
            "ground_truth":      chosen_gt,
            "correct":           correct,
            "conviction":        conviction,
            "alpha_pct":         round(alpha * 100, 3),
            "chosen_return_pct": round(chosen_return * 100, 4),
            "sector_avg_pct":    round(sector_avg * 100, 4),
            "virtual_capital":   round(self.virtual_capital, 6),
            "drawdown":          round(drawdown, 4),
            "early_stop":        early_stop,
        }

        if not done:
            self.current_obs = self._build_obs(self.current_step_idx)
        else:
            self.episodes_completed += 1
            self.current_obs = None

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
        "status":       "ok",
        "env":          "IndicatorsEnv",
        "version":      "4.0.0",
        "tasks":        len(TASKS),
        "episode_steps": TASK_MAX_STEPS,
        "step_spacing":  STEP_SPACING,
        "sectors":       list(SECTOR_GROUPS.keys()),
    }


@app.get("/tasks")
def get_tasks():
    """Returns all task definitions, action schemas, and episode lengths."""
    return {
        "tasks":         TASKS,
        "total":         len(TASKS),
        "action_schema": {
            "stock":     {"type": "string", "description": "NSE symbol or 'NONE' to skip"},
            "direction": {"type": "string", "enum": ["Bullish", "Bearish", "NONE"]},
            "conviction":{"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "episode_steps":  TASK_MAX_STEPS,
        "step_spacing":   STEP_SPACING,
    }


@app.post("/reset", response_model=ResetResult)
def reset(session_id: Optional[str] = None, term: str = "medium"):
    sess   = _get_or_create(session_id, term=term)
    result = sess.reset()
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Could not fetch market data. Retry or try a different term.",
        )
    return result


@app.post("/step", response_model=StepResult)
def step(action: MultiStockAction, session_id: Optional[str] = None):
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
    episode_results: list of {predicted, ground_truth, conviction}.
    Returns score in (0.0, 1.0).
    """
    if request.task_id not in TASK_BY_ID:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id: {request.task_id}. "
                   f"Valid: {list(TASK_BY_ID.keys())}",
        )
    return grade_task(request.task_id, request.episode_results)


@app.get("/baseline")
def baseline():
    """
    Run the built-in random baseline across all 3 tasks (10 episodes each).
    Expected: overall_mean ≈ 0.33 (random agent on 3-class directional problem).
    short  : 5×10=50  steps
    medium : 10×10=100 steps
    long   : ≤15×10=150 steps (may be fewer with early termination)
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
                # Random agent: random stock, random direction (including NONE)
                available = reset_result.observation.available_stocks
                if hasattr(sess.current_obs, "available_stocks") and sess.current_obs:
                    available = sess.current_obs.available_stocks
                chosen_stock = random.choice(available + ["NONE"])
                direction    = "NONE" if chosen_stock == "NONE" else random.choice(["Bullish", "Bearish"])
                action = MultiStockAction(
                    stock      = chosen_stock,
                    direction  = direction,
                    conviction = round(random.uniform(0.3, 0.9), 2),
                )
                step_result = sess.step(action)
                episode_results.append({
                    "predicted":    direction,
                    "ground_truth": step_result.info.get("ground_truth", "N/A"),
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
        "agent":        "random_baseline",
        "seed":         42,
        "tasks":        results,
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
                action = MultiStockAction(**msg.get("action", {}))
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
