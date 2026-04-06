"""
indicators_env_client.py — Typed sync/async client for IndicatorsEnv.

Mirrors the openenv-core EnvClient pattern so it works seamlessly with
TRL's GRPOTrainer and the openenv course module interface.

Usage (sync):
    from indicators_env_client import IndicatorsEnvClient, IndicatorsAction
    with IndicatorsEnvClient(base_url="http://localhost:8000").sync() as env:
        obs = env.reset()
        result = env.step(IndicatorsAction(direction="Bullish", conviction=0.8))
        print(result.reward)

Usage (async):
    async with IndicatorsEnvClient(base_url="http://localhost:8000") as env:
        obs = await env.reset()
        result = await env.step(IndicatorsAction(direction="Bullish", conviction=0.8))
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
import websockets


# ─── Data classes (mirror server schemas) ────────────────────────────────────

@dataclass
class IndicatorsAction:
    direction: str    # "Bullish" | "Bearish" | "Neutral"
    conviction: float = 0.5


@dataclass
class IndicatorsObservation:
    symbol: str
    date: str
    term: str
    current_price: float
    indicators: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IndicatorsObservation":
        return cls(
            symbol=d["symbol"],
            date=d["date"],
            term=d["term"],
            current_price=d["current_price"],
            indicators=d["indicators"],
        )

    def to_prompt(self) -> str:
        """Render observation as a structured LLM prompt string."""
        ind = self.indicators
        ma  = ind.get("moving_averages", {})
        rsi = ind.get("rsi", {})
        mac = ind.get("macd", {})
        bb  = ind.get("bollinger_bands", {})
        adx = ind.get("adx", {})
        vol = ind.get("enhanced_volume", {})
        vlt = ind.get("volatility", {})
        sto = ind.get("stochastic", {})
        piv = ind.get("pivot_points", {})

        return f"""[TERM: {self.term}]
Stock: {self.symbol} | Date: {self.date} | Price: {self.current_price}

MOVING AVERAGES
  SMA20={ma.get('sma_20')} | SMA50={ma.get('sma_50')} | SMA200={ma.get('sma_200')}
  EMA20={ma.get('ema_20')} | EMA50={ma.get('ema_50')}
  Signal: {ma.get('signal')} | Golden Cross: {ma.get('golden_cross')} | Death Cross: {ma.get('death_cross')}

MOMENTUM
  RSI14={rsi.get('rsi_14')} ({rsi.get('status')}) | Trend: {rsi.get('trend')}
  MACD={mac.get('macd_line')} | Signal={mac.get('signal_line')} | Hist={mac.get('histogram')} | {mac.get('signal')} | Crossover: {mac.get('crossover')}
  Stochastic K={sto.get('k')} D={sto.get('d')} ({sto.get('signal')})

VOLATILITY / BANDS
  BB Upper={bb.get('upper')} | Mid={bb.get('middle')} | Lower={bb.get('lower')}
  %B={bb.get('percent_b')} | Bandwidth={bb.get('bandwidth')} | Squeeze={bb.get('squeeze')}
  ATR={vlt.get('atr_14')} | Vol Regime={vlt.get('regime')} | Vol Ratio={vlt.get('volatility_ratio')}

TREND
  ADX={adx.get('adx')} ({adx.get('trend_strength')}) | +DI={adx.get('plus_di')} | -DI={adx.get('minus_di')} | Direction: {adx.get('trend_direction')}

VOLUME
  VWAP={vol.get('vwap')} | Price vs VWAP={vol.get('price_vs_vwap_pct')}%
  MFI={vol.get('mfi')} ({vol.get('mfi_status')}) | CMF={vol.get('cmf')} ({vol.get('cmf_signal')})
  OBV Trend={ind.get('volume', {}).get('obv_trend')} | Volume Ratio={ind.get('volume', {}).get('volume_ratio')}x
  A/D Line Trend={vol.get('ad_line_trend')}

PIVOT POINTS
  R2={piv.get('r2')} | R1={piv.get('r1')} | Pivot={piv.get('pivot')} | S1={piv.get('s1')} | S2={piv.get('s2')}

Based on the above technical indicators, predict the {self.term}-term direction.
Respond ONLY with a JSON object in this exact format:
{{"direction": "Bullish" | "Bearish" | "Neutral", "conviction": <float 0.0-1.0>}}"""


@dataclass
class StepResult:
    observation: Optional[IndicatorsObservation]
    reward: float
    done: bool
    info: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StepResult":
        obs_d = d.get("observation")
        return cls(
            observation=IndicatorsObservation.from_dict(obs_d) if obs_d else None,
            reward=d["reward"],
            done=d["done"],
            info=d.get("info", {}),
        )


@dataclass
class ResetResult:
    observation: IndicatorsObservation
    info: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResetResult":
        return cls(
            observation=IndicatorsObservation.from_dict(d["observation"]),
            info=d.get("info", {}),
        )


# ─── Async client ─────────────────────────────────────────────────────────────

class IndicatorsEnvClient:
    """Async client for IndicatorsEnv using WebSocket for step() calls."""

    def __init__(self, base_url: str = "http://localhost:8000", term: str = "medium"):
        self.base_url = base_url.rstrip("/")
        self.term = term
        self._ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self._ws = None
        self._http: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        self._ws = await websockets.connect(f"{self._ws_url}?term={self.term}")
        return self

    async def __aexit__(self, *args):
        if self._ws:
            await self._ws.close()
        if self._http:
            await self._http.aclose()

    async def reset(self) -> ResetResult:
        await self._ws.send(json.dumps({"method": "reset"}))
        raw = await self._ws.recv()
        return ResetResult.from_dict(json.loads(raw))

    async def step(self, action: IndicatorsAction) -> StepResult:
        await self._ws.send(json.dumps({
            "method": "step",
            "action": {"direction": action.direction, "conviction": action.conviction},
        }))
        raw = await self._ws.recv()
        return StepResult.from_dict(json.loads(raw))

    async def state(self) -> Dict[str, Any]:
        await self._ws.send(json.dumps({"method": "state"}))
        raw = await self._ws.recv()
        return json.loads(raw)

    def sync(self) -> "_SyncWrapper":
        return _SyncWrapper(self)


class _SyncWrapper:
    """Synchronous context manager wrapper (mirrors openenv-core .sync() pattern)."""

    def __init__(self, client: IndicatorsEnvClient):
        self._client = client
        self._loop = asyncio.new_event_loop()

    def __enter__(self):
        self._loop.run_until_complete(self._client.__aenter__())
        return self

    def __exit__(self, *args):
        self._loop.run_until_complete(self._client.__aexit__(*args))
        self._loop.close()

    def reset(self) -> ResetResult:
        return self._loop.run_until_complete(self._client.reset())

    def step(self, action: IndicatorsAction) -> StepResult:
        return self._loop.run_until_complete(self._client.step(action))

    def state(self) -> Dict[str, Any]:
        return self._loop.run_until_complete(self._client.state())
