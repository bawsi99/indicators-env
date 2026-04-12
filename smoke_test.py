"""
smoke_test.py — Local smoke test for IndicatorsEnv v4.1

Two modes:
  Unit mode (default) — imports env code directly, uses synthetic data.
                        Runs in <5s, no network needed, no yfinance calls.
  Live mode (--live)  — starts the HTTP server, tests real yfinance data.
                        Requires network access to Yahoo Finance.

Usage:
  python smoke_test.py           # unit mode (fast, offline)
  python smoke_test.py --live    # full integration test (requires network)
"""

import sys
import os
import argparse
import random

HACKATHON_DIR = os.path.dirname(__file__)
ENV_DIR = os.path.join(HACKATHON_DIR, "env")
sys.path.insert(0, ENV_DIR)

PASS = "✅"
FAIL = "❌"
failures = 0


def check(label, condition, detail=""):
    global failures
    status = PASS if condition else FAIL
    print(f"  {status}  {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures += 1
    return condition


def section(title):
    print(f"\n{'─'*58}")
    print(f"  {title}")
    print(f"{'─'*58}")


# ── Build a synthetic episode (no yfinance) ───────────────────────────────────

def make_fake_indicators():
    return {
        "moving_averages": {"sma_20": 1600.0, "sma_50": 1580.0, "signal": "bullish"},
        "rsi":  {"rsi_14": 58.3, "trend": "up", "status": "neutral", "signal": "neutral"},
        "macd": {"macd_line": 12.4, "signal_line": 9.1, "histogram": 3.3, "signal": "bullish"},
        "bollinger_bands": {"upper": 1680.0, "lower": 1540.0, "percent_b": 0.62, "signal": "neutral"},
        "adx":  {"adx_14": 28.5, "plus_di": 22.1, "minus_di": 14.3, "trend_strength": "moderate"},
        "stochastic": {"percent_k": 64.2, "percent_d": 58.1, "signal": "neutral"},
        "volatility": {"atr_14": 28.3, "volatility_regime": "normal"},
        "enhanced_volume": {"obv": 1234567, "mfi_14": 55.2, "cmf_20": 0.12, "volume_ratio": 1.05},
        "pivot_points": {"pivot": 1625.0, "r1": 1660.0, "s1": 1590.0},
    }


def make_fake_step(step_idx, symbols, term, spacing_days):
    """Build one synthetic episode step for 3 stocks."""
    base_date = f"2022-{3 + step_idx:02d}-15"
    step_stocks = []
    for i, sym in enumerate(symbols):
        base_return = random.uniform(-0.04, 0.04)
        step_stocks.append({
            "symbol": sym,
            "obs_dict": {
                "symbol":             sym,
                "date":               base_date,
                "term":               term.upper(),
                "current_price":      round(1500 + i * 100 + step_idx * 10, 2),
                "rsi_14":             round(45 + i * 5 + step_idx * 2, 2),
                "rsi_trend":          "up" if step_idx % 2 == 0 else "down",
                "price_momentum_pct": round(step_idx * 0.5 + i * 0.3, 3),
                "indicators":         make_fake_indicators(),
            },
            "gt":                   random.choice(["Bullish", "Bearish", "Neutral"]),
            "actual_period_return": base_return,
        })
    return {
        "step_index": step_idx,
        "step_date":  base_date,
        "stocks":     step_stocks,
        "macro":      None,
    }


def make_fake_episode(symbols, n_steps, term):
    spacing_days = {"short": 1, "medium": 5, "long": 20}.get(term, 5)
    return [make_fake_step(i, symbols, term, spacing_days) for i in range(n_steps)]


# ─────────────────────────────────────────────────────────────────────────────
#  UNIT MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_unit_tests():
    print("\nMode: UNIT (synthetic data, no yfinance)\n")

    # Import env modules directly
    from indicators_env import (
        EnvSession, MultiStockAction, grade_task,
        TASK_MAX_STEPS, SECTOR_GROUPS,
        TRANSACTION_COST, DRAWDOWN_LIMITS,
    )

    SYMBOLS = ["HDFCBANK", "ICICIBANK", "AXISBANK"]

    # ── 1. Constants & imports ────────────────────────────────────────────────
    section("1. Constants & imports")
    check("TASK_MAX_STEPS short=5",    TASK_MAX_STEPS["short"]  == 5,  str(TASK_MAX_STEPS))
    check("TASK_MAX_STEPS medium=10",  TASK_MAX_STEPS["medium"] == 10, str(TASK_MAX_STEPS))
    check("TASK_MAX_STEPS long=15",    TASK_MAX_STEPS["long"]   == 15, str(TASK_MAX_STEPS))
    check("5 sectors",                 len(SECTOR_GROUPS) == 5,        str(list(SECTOR_GROUPS.keys())))
    check("each sector has 5 stocks",  all(len(v) == 5 for v in SECTOR_GROUPS.values()))
    # v4.1 constants
    check("TRANSACTION_COST=0.001",    TRANSACTION_COST == 0.001,      str(TRANSACTION_COST))
    check("DRAWDOWN_LIMITS short=5%",  DRAWDOWN_LIMITS["short"]  == 0.05)
    check("DRAWDOWN_LIMITS medium=10%",DRAWDOWN_LIMITS["medium"] == 0.10)
    check("DRAWDOWN_LIMITS long=15%",  DRAWDOWN_LIMITS["long"]   == 0.15)

    # ── 2. Multi-step episode structure ───────────────────────────────────────
    section("2. Multi-step episode structure (short, 5 steps, synthetic)")
    sess = EnvSession(session_id="unit-short", term="short")
    sess.episode_steps = make_fake_episode(SYMBOLS, 5, "short")
    sess.sector        = "banking"
    sess.symbols       = SYMBOLS
    sess.current_obs   = sess._build_obs(0)

    obs = sess.current_obs
    check("step=1",                        obs.step == 1,              str(obs.step))
    check("max_steps=5",                   obs.max_steps == 5,         str(obs.max_steps))
    check("sector=banking",                obs.sector == "banking",    obs.sector)
    check("available_stocks has 3",        len(obs.available_stocks) == 3)
    check("stocks dict has 3 entries",     len(obs.stocks) == 3)
    check("signal_history empty on step1", obs.signal_history == [])

    print("\n  Steps:")
    for i in range(1, 6):
        result = sess.step(MultiStockAction(
            stock="HDFCBANK", direction="Bullish", conviction=0.7
        ))
        expect_done = (i == 5)
        ok = result.done == expect_done
        if not ok:
            failures += 1
        print(f"    step={result.info['step']}  done={result.done}  "
              f"reward={result.reward:+.4f}  "
              f"alpha={result.info.get('alpha_pct',0):+.3f}%  "
              f"GT={result.info.get('ground_truth','?')}  "
              f"{'✅' if ok else '❌ expected done='+str(expect_done)}")

    # ── 3. NONE pass action ───────────────────────────────────────────────────
    section("3. NONE pass action")
    sess2 = EnvSession(session_id="unit-none", term="short")
    sess2.episode_steps = make_fake_episode(SYMBOLS, 5, "short")
    sess2.sector        = "banking"
    sess2.symbols       = SYMBOLS
    sess2.current_obs   = sess2._build_obs(0)

    r_none = sess2.step(MultiStockAction(stock="NONE", direction="NONE", conviction=0.0))
    check("NONE reward=0.0",          r_none.reward == 0.0,                            str(r_none.reward))
    check("NONE done=False",          r_none.done == False,                            str(r_none.done))
    check("NONE ground_truth=N/A",    r_none.info.get("ground_truth") == "N/A",        str(r_none.info.get("ground_truth")))
    check("NONE chosen_stock=NONE",   r_none.info.get("chosen_stock") == "NONE",       str(r_none.info.get("chosen_stock")))
    check("NONE alpha_pct=0.0",       r_none.info.get("alpha_pct") == 0.0,             str(r_none.info.get("alpha_pct")))

    # signal_history entry for NONE step
    r_none2 = sess2.step(MultiStockAction(stock="HDFCBANK", direction="Bullish", conviction=0.7))
    hist = r_none2.observation.signal_history if r_none2.observation else []
    check("signal_history len=2 after step 2",  len(hist) == 2, str(len(hist)))
    check("step1 in history is NONE",           hist[0]["direction"] == "NONE" if hist else False)

    # ── 4. Relative alpha reward ──────────────────────────────────────────────
    section("4. Relative alpha reward formula")
    # Inject a step where we know exact returns
    sess3 = EnvSession(session_id="unit-alpha", term="short")
    fake_ep = make_fake_episode(SYMBOLS, 5, "short")
    # Force known returns: HDFCBANK +4%, ICICIBANK +1%, AXISBANK +1%
    # sector_avg = (0.04 + 0.01 + 0.01) / 3 = 0.02
    # alpha for HDFCBANK = 0.04 - 0.02 = 0.02
    # reward = 0.02 × +1 × 0.8 × 50 = 0.80
    fake_ep[0]["stocks"][0]["actual_period_return"] = 0.04   # HDFCBANK
    fake_ep[0]["stocks"][1]["actual_period_return"] = 0.01   # ICICIBANK
    fake_ep[0]["stocks"][2]["actual_period_return"] = 0.01   # AXISBANK
    sess3.episode_steps = fake_ep
    sess3.sector        = "banking"
    sess3.symbols       = SYMBOLS
    sess3.current_obs   = sess3._build_obs(0)

    r_alpha = sess3.step(MultiStockAction(stock="HDFCBANK", direction="Bullish", conviction=0.8))
    expected_reward = round((0.04 - 0.02) * 1 * 0.8 * 50, 4)   # = 0.8
    check(f"alpha reward ≈ {expected_reward}",
          abs(r_alpha.reward - expected_reward) < 0.01,
          f"got {r_alpha.reward:.4f}")
    check(f"alpha_pct ≈ 2.0%",
          abs(r_alpha.info.get("alpha_pct", 0) - 2.0) < 0.1,
          f"got {r_alpha.info.get('alpha_pct',0):.3f}%")

    # Bearish on underperformer: AXISBANK vs sector avg
    # AXISBANK = +1%, avg = +2% → alpha = -1% → Bearish × -1 × 0.6 × 50 = +0.30
    fake_ep2 = make_fake_episode(SYMBOLS, 5, "short")
    fake_ep2[0]["stocks"][0]["actual_period_return"] = 0.04
    fake_ep2[0]["stocks"][1]["actual_period_return"] = 0.04
    fake_ep2[0]["stocks"][2]["actual_period_return"] = 0.01   # underperformer
    sess4 = EnvSession(session_id="unit-alpha2", term="short")
    sess4.episode_steps = fake_ep2
    sess4.sector = "banking"
    sess4.symbols = SYMBOLS
    sess4.current_obs = sess4._build_obs(0)
    r_bear = sess4.step(MultiStockAction(stock="AXISBANK", direction="Bearish", conviction=0.6))
    # sector_avg = (0.04+0.04+0.01)/3 = 0.03; alpha = 0.01-0.03 = -0.02
    # reward = -0.02 × -1 × 0.6 × 50 = +0.60
    expected_bear = round((-0.02) * -1 * 0.6 * 50, 4)
    check(f"bearish underperformer reward ≈ {expected_bear}",
          abs(r_bear.reward - expected_bear) < 0.01,
          f"got {r_bear.reward:.4f}")

    # ── 5. Grader ─────────────────────────────────────────────────────────────
    section("5. Grader")

    # short task: 2 correct out of 3 active, 2 NONE
    ep = [
        {"predicted": "Bullish", "ground_truth": "Bullish", "conviction": 0.7},
        {"predicted": "NONE",    "ground_truth": "N/A",     "conviction": 0.0},
        {"predicted": "Bearish", "ground_truth": "Bearish", "conviction": 0.8},
        {"predicted": "NONE",    "ground_truth": "N/A",     "conviction": 0.0},
        {"predicted": "Bullish", "ground_truth": "Neutral",  "conviction": 0.5},
    ]
    g = grade_task("short_term_direction", ep)
    check("short: score > 0",       g.score > 0,        f"score={g.score}")
    check("short: active_steps=3",  g.breakdown["active_steps"] == 3)
    check("short: correct=2",       g.breakdown["correct"] == 2)
    check("short: score ≈ 0.667",   abs(g.score - 0.667) < 0.01, f"score={g.score:.4f}")

    # all NONE → 0.001
    g2 = grade_task("short_term_direction", [
        {"predicted": "NONE", "ground_truth": "N/A", "conviction": 0.0},
        {"predicted": "NONE", "ground_truth": "N/A", "conviction": 0.0},
    ])
    check("all-NONE → score=0.001", g2.score == 0.001, f"score={g2.score}")

    # medium: participation bonus
    med_ep = [
        {"predicted": "Bullish", "ground_truth": "Bullish", "conviction": 0.7},
        {"predicted": "Bearish", "ground_truth": "Bearish", "conviction": 0.8},
        {"predicted": "NONE",    "ground_truth": "N/A",     "conviction": 0.0},
    ]
    g3 = grade_task("medium_term_direction", med_ep)
    check("medium: score > 0",       g3.score > 0,  f"score={g3.score:.4f}")
    check("medium: active_steps=2",  g3.breakdown["active_steps"] == 2)

    # long: conviction calibration
    long_ep = [
        {"predicted": "Bullish", "ground_truth": "Bullish", "conviction": 0.8},  # 1.0
        {"predicted": "Bullish", "ground_truth": "Bullish", "conviction": 0.5},  # 0.5
        {"predicted": "Bearish", "ground_truth": "Bullish", "conviction": 0.9},  # -0.1
    ]
    g4 = grade_task("long_term_conviction", long_ep)
    check("long: score > 0",        g4.score > 0,  f"score={g4.score:.4f}")
    check("long: active_steps=3",   g4.breakdown["active_steps"] == 3)

    # ── 6. Drawdown limit — long task (15%) ──────────────────────────────────
    section("6. Early termination on drawdown (long task, 15% limit)")
    sess5 = EnvSession(session_id="unit-dd", term="long")
    fake_long = make_fake_episode(SYMBOLS, 15, "long")

    # Force HDFCBANK (index 0) to massively underperform sector average.
    # HDFCBANK=-0.15, ICICIBANK=0.0, AXISBANK=0.0 → avg=-0.05
    # alpha = -0.15 - (-0.05) = -0.10 per step
    # Bullish pick: pnl = -0.10 × 1 × 0.8 = -0.08 → drawdown >15% after 2 steps.
    for step_d in fake_long:
        step_d["stocks"][0]["actual_period_return"] = -0.15   # HDFCBANK underperforms
        step_d["stocks"][1]["actual_period_return"] =  0.0    # ICICIBANK flat
        step_d["stocks"][2]["actual_period_return"] =  0.0    # AXISBANK flat

    sess5.episode_steps = fake_long
    sess5.sector  = "banking"
    sess5.symbols = SYMBOLS
    sess5.current_obs = sess5._build_obs(0)

    early_terminated = False
    for _ in range(15):
        r = sess5.step(MultiStockAction(stock="HDFCBANK", direction="Bullish", conviction=0.8))
        if r.done:
            early_terminated = True
            dd = r.info.get("drawdown", 0)
            dl = r.info.get("drawdown_limit", DRAWDOWN_LIMITS["long"])
            check(f"early termination triggered (drawdown={dd:.3f} > limit={dl})",
                  dd > dl, f"step={r.info.get('step')}")
            break
    check("drawdown limit terminates before step 15", early_terminated)

    # ── 7. v4.1: capital updates on active step ───────────────────────────────
    section("7. v4.1 — Capital tracks cumulative alpha P&L (all tasks)")
    sess7 = EnvSession(session_id="unit-cap", term="short")
    fake_cap = make_fake_episode(SYMBOLS, 5, "short")
    # HDFCBANK +4%, others +1% → avg = (0.04+0.01+0.01)/3 ≈ 0.02
    # alpha = 0.04 - 0.02 = 0.02; pnl = 0.02 × 1 × 0.7 = 0.014
    # capital after = 1.0 + 0.014 = 1.014
    fake_cap[0]["stocks"][0]["actual_period_return"] = 0.04
    fake_cap[0]["stocks"][1]["actual_period_return"] = 0.01
    fake_cap[0]["stocks"][2]["actual_period_return"] = 0.01
    sess7.episode_steps = fake_cap
    sess7.sector  = "banking"
    sess7.symbols = SYMBOLS
    sess7.current_obs = sess7._build_obs(0)

    check("capital=1.0 before any step", sess7.virtual_capital == 1.0)
    r7 = sess7.step(MultiStockAction(stock="HDFCBANK", direction="Bullish", conviction=0.7))
    expected_pnl = round(0.02 * 0.7, 6)    # alpha × conviction
    expected_cap = round(1.0 + expected_pnl, 6)
    check(f"capital ≈ {expected_cap} after profitable step",
          abs(sess7.virtual_capital - expected_cap) < 0.001,
          f"got {sess7.virtual_capital:.6f}")
    check("obs.capital matches internal capital",
          abs(r7.observation.capital - sess7.virtual_capital) < 1e-6 if r7.observation else False,
          f"obs.capital={r7.observation.capital if r7.observation else 'N/A'}")
    check("obs.current_holding=HDFCBANK after pick",
          r7.observation.current_holding == "HDFCBANK" if r7.observation else False)

    # ── 8. v4.1: transaction cost on holding switch ───────────────────────────
    section("8. v4.1 — Transaction cost when switching holdings")
    sess8 = EnvSession(session_id="unit-tx", term="medium")
    fake_tx = make_fake_episode(SYMBOLS, 10, "medium")
    for s_step in fake_tx:
        s_step["stocks"][0]["actual_period_return"] = 0.02   # HDFCBANK
        s_step["stocks"][1]["actual_period_return"] = 0.01   # ICICIBANK
        s_step["stocks"][2]["actual_period_return"] = 0.01   # AXISBANK
    sess8.episode_steps = fake_tx
    sess8.sector  = "banking"
    sess8.symbols = SYMBOLS
    sess8.current_obs = sess8._build_obs(0)

    # Step 1: pick HDFCBANK — no prior holding, tx_cost should be 0
    r8a = sess8.step(MultiStockAction(stock="HDFCBANK", direction="Bullish", conviction=0.7))
    check("tx_cost=0 on first pick (no prior holding)",
          r8a.info.get("tx_cost", -1) == 0.0,
          f"tx_cost={r8a.info.get('tx_cost')}")
    check("current_holding=HDFCBANK after step 1",
          r8a.info.get("current_holding") == "HDFCBANK")

    # Step 2: switch to ICICIBANK — tx_cost = 0.001 × capital_after_step1
    cap_after_step1 = sess8.virtual_capital
    r8b = sess8.step(MultiStockAction(stock="ICICIBANK", direction="Bullish", conviction=0.7))
    expected_tx = round(TRANSACTION_COST * cap_after_step1, 6)
    check(f"tx_cost ≈ {expected_tx:.6f} on switch HDFCBANK→ICICIBANK",
          abs(r8b.info.get("tx_cost", -1) - expected_tx) < 1e-6,
          f"got {r8b.info.get('tx_cost')}")
    check("current_holding=ICICIBANK after switch",
          r8b.info.get("current_holding") == "ICICIBANK")

    # Step 3: same stock (ICICIBANK→ICICIBANK) — no tx_cost
    r8c = sess8.step(MultiStockAction(stock="ICICIBANK", direction="Bullish", conviction=0.7))
    check("tx_cost=0 when staying in same holding",
          r8c.info.get("tx_cost", -1) == 0.0,
          f"tx_cost={r8c.info.get('tx_cost')}")

    # Step 4: NONE pass — no tx_cost, holding preserved
    r8d = sess8.step(MultiStockAction(stock="NONE", direction="NONE", conviction=0.0))
    check("tx_cost=0 on NONE pass",
          r8d.info.get("tx_cost", -1) == 0.0,
          f"tx_cost={r8d.info.get('tx_cost')}")
    check("holding=ICICIBANK preserved after NONE pass",
          r8d.info.get("current_holding") == "ICICIBANK")

    # ── 9. v4.1: per-task drawdown limits ────────────────────────────────────
    section("9. v4.1 — Per-task drawdown limits (short=5%, medium=10%, long=15%)")
    # Manually induce 6% drawdown and test short (limit 5%) vs medium (limit 10%)
    for term_t, limit, expect_done in [("short", 0.05, True), ("medium", 0.10, False)]:
        sess9 = EnvSession(session_id=f"unit-perlimit-{term_t}", term=term_t)
        n = TASK_MAX_STEPS[term_t]
        fake9 = make_fake_episode(SYMBOLS, n, term_t)
        # HDFCBANK underperforms: -0.09, others 0.0 → avg = -0.03
        # alpha = -0.09 - (-0.03) = -0.06; pnl = -0.06 × 1 × 1.0 = -0.06
        # After 1 step: capital = 0.94, drawdown = 6%
        for s_step in fake9:
            s_step["stocks"][0]["actual_period_return"] = -0.09
            s_step["stocks"][1]["actual_period_return"] = 0.0
            s_step["stocks"][2]["actual_period_return"] = 0.0
        sess9.episode_steps = fake9
        sess9.sector  = "banking"
        sess9.symbols = SYMBOLS
        sess9.current_obs = sess9._build_obs(0)

        r9 = sess9.step(MultiStockAction(stock="HDFCBANK", direction="Bullish", conviction=1.0))
        dd = r9.info.get("drawdown", 0)
        dl = r9.info.get("drawdown_limit", limit)
        check(f"{term_t}: drawdown={dd:.3f} > limit={dl} → done={expect_done}",
              r9.done == expect_done,
              f"done={r9.done}, drawdown={dd:.3f}, limit={dl}")


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_live_tests():
    import subprocess
    import time
    import requests

    BASE = "http://localhost:7860"
    PYTHON = sys.executable

    print("\nMode: LIVE (real yfinance + HTTP server)\n")

    # Start server
    proc = subprocess.Popen(
        [PYTHON, "indicators_env.py"],
        cwd=ENV_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print("Server starting...", end="", flush=True)
    for _ in range(30):
        time.sleep(1)
        print(".", end="", flush=True)
        try:
            if requests.get(f"{BASE}/health", timeout=2).status_code == 200:
                print(" ready.\n")
                break
        except Exception:
            pass
    else:
        print(" TIMEOUT")
        proc.terminate()
        return

    try:
        section("L1. Health")
        h = requests.get(f"{BASE}/health", timeout=5).json()
        check("version=4.1.0",         h.get("version") == "4.1.0",      h.get("version"))
        check("drawdown_limits present", "drawdown_limits" in h,          str(h.get("drawdown_limits")))
        check("transaction_cost=0.001",  h.get("transaction_cost") == 0.001)
        print(f"  episode_steps={h.get('episode_steps')}  step_spacing={h.get('step_spacing')}")

        section("L2. Reset + 5-step short episode")
        r = requests.post(f"{BASE}/reset", params={"term": "short"}, timeout=90)
        check("reset 200", r.status_code == 200, str(r.status_code))
        if r.status_code != 200:
            print(f"  {r.text}")
            return

        data = r.json()
        sid  = data["info"]["session_id"]
        obs  = data["observation"]
        check("sector present",        bool(obs["sector"]),              obs["sector"])
        check("3 stocks",              len(obs["available_stocks"]) == 3)
        print(f"\n  sector={obs['sector']}  stocks={obs['available_stocks']}")

        chosen = obs["available_stocks"][0]
        print("\n  Steps:")
        for i in range(1, 6):
            s = requests.post(
                f"{BASE}/step",
                params={"session_id": sid},
                json={"stock": chosen, "direction": "Bullish", "conviction": 0.7},
                timeout=30,
            ).json()
            ok = s["done"] == (i == 5)
            if not ok:
                global failures
                failures += 1
            print(f"    step={s['info']['step']}  done={s['done']}  "
                  f"reward={s['reward']:+.4f}  "
                  f"alpha={s['info'].get('alpha_pct',0):+.3f}%  "
                  f"GT={s['info'].get('ground_truth','?')}  "
                  f"{'✅' if ok else '❌'}")

        section("L3. Baseline (~60–120s)")
        print("  Running...", end="", flush=True)
        b = requests.get(f"{BASE}/baseline", timeout=600).json()
        print(" done.")
        check("short ≥ 50 steps",  b["tasks"]["short_term_direction"]["num_episodes"] >= 50)
        check("overall_mean > 0",  b["overall_mean"] > 0, f"{b['overall_mean']:.4f}")
        print(f"  overall_mean={b['overall_mean']:.4f}")

    finally:
        proc.terminate()
        proc.wait(timeout=5)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run live HTTP tests (requires network)")
    args = parser.parse_args()

    if args.live:
        run_live_tests()
    else:
        run_unit_tests()

    print(f"\n{'═'*58}")
    if failures == 0:
        print(f"  {PASS}  ALL CHECKS PASSED")
    else:
        print(f"  {FAIL}  {failures} CHECK(S) FAILED")
    print(f"{'═'*58}\n")
    sys.exit(0 if failures == 0 else 1)
