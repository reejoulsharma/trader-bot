"""
decision/decision_engine.py — Rule-based decision layer.

Combines a symbol's recent signals into a BUY/SELL/HOLD recommendation
using a fixed weight/threshold model. Read-only and side-effect free: it
computes a Decision, it never places an order.

Day-trading rule: positions must never carry overnight. Two independent
mechanisms enforce this:
  - Conviction on the BUY side decays to zero approaching ENTRY_CUTOFF_TIME,
    and BUY is disallowed entirely outside the 9:15-15:00 IST window.
  - must_flatten is True whenever `now` falls outside the 9:15-15:15 IST
    safe window — any position still held must be closed immediately,
    independent of score. This is robust to being checked at any time
    (before open, after close, the next morning), not just during the
    trading day.

Known gap: this module has no notion of whether a position is actually
held. `action` is the score-driven recommendation; `must_flatten` is the
time-driven override. A future execution layer must combine both with
real portfolio state — must_flatten means "close it if you hold it," not
"a position exists to close."

Run with:  python -m decision.decision_engine
"""

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from config import settings
from storage.timescale import get_db_pool, get_recent_signals

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

IST = ZoneInfo("Asia/Kolkata")

# Directional weights, applied per distinct (signal_type, value) — only the
# most recent occurrence of each within the lookback window counts, so a
# persistent condition re-detected every engine cycle doesn't dominate the
# score just because it kept getting re-observed.
#
# Recalibrated from evaluation.signal_evaluation output run against ~1 week
# of live signals (see git history for the exact run). Original weights
# were educated guesses made before any real data existed; these replace
# them with what the data actually showed:
#   - sentiment:positive flipped bullish -> bearish (-1.5, was +1.0): 24-28%
#     win rate, consistently negative mean return at every horizon (5/15/60
#     min). Small sample (n=29) but directionally consistent across all
#     three independent horizons, which is what makes it trustworthy
#     despite the modest count — noise doesn't usually agree with itself
#     three times in a row.
#   - pattern:support flipped bullish -> bearish (-0.75, was +0.75): ~41%
#     win rate, consistently negative, at every horizon, backed by a huge
#     sample (~4,000) — a milder edge than sentiment but very high
#     confidence it's real given the sample size.
#   - pattern:breakout stays bullish but reduced (+1.0, was +1.5): genuine
#     edge at 5-15 min (55-67% win rate) fading to worse-than-random by 60
#     min, on a small sample (n~27) — real but not strong evidence.
#   - sentiment:negative stays bearish but heavily discounted (-0.25, was
#     -1.0): win rate bounces both sides of 50% across horizons
#     (37.5/62.5/40.0), small sample (n~16) — no real evidence either way,
#     kept as a weak prior rather than zeroed out entirely.
#
# Practical consequence: pattern:breakout is now the only bullish-weighted
# signal, and its solo weight no longer crosses BUY_THRESHOLD alone — BUY
# effectively requires breakout plus a volume anomaly together. This is
# intentional, not a bug: the data doesn't support confident BUY calls
# right now, so the model shouldn't produce them just to seem balanced.
SIGNAL_WEIGHTS: dict[tuple[str, str], float] = {
    ("sentiment", "positive"): -1.5,
    ("sentiment", "negative"): -0.25,
    ("pattern", "breakout"): 1.0,
    ("pattern", "support"): -0.75,
}

# volume:anomaly has no direction — it amplifies whatever directional score
# already exists rather than contributing its own weight.
VOLUME_ANOMALY_MULTIPLIER = 1.25

LOOKBACK_MINUTES = 30

BUY_THRESHOLD = 1.2
SELL_THRESHOLD = -1.2

# Day-trading enforcement, all IST.
MARKET_OPEN_TIME = time(9, 15)
DECAY_START_TIME = time(14, 0)   # BUY conviction starts decaying
ENTRY_CUTOFF_TIME = time(15, 0)  # no new BUYs at/after this
FORCE_FLATTEN_TIME = time(15, 15)  # must_flatten becomes True at/after this


@dataclass
class Decision:
    symbol: str
    raw_score: float
    decayed_score: float
    action: str  # "BUY" | "SELL" | "HOLD"
    must_flatten: bool
    contributing_signals: list[str] = field(default_factory=list)


def _latest_per_type_value(signals: list[dict]) -> list[dict]:
    """Collapse repeated (signal_type, value) pairs to their most recent occurrence."""
    latest: dict[tuple[str, str], dict] = {}
    for sig in signals:
        key = (sig["signal_type"], sig["value"])
        if key not in latest or sig["time"] > latest[key]["time"]:
            latest[key] = sig
    return list(latest.values())


def compute_raw_score(signals: list[dict]) -> tuple[float, list[str]]:
    """
    Combine recent signals into a single directional score.
    Returns (score, human-readable contributing signal descriptions).
    """
    deduped = _latest_per_type_value(signals)

    directional_score = 0.0
    has_volume_anomaly = False
    contributions = []

    for sig in deduped:
        key = (sig["signal_type"], sig["value"])
        if key == ("volume", "anomaly"):
            has_volume_anomaly = True
            contributions.append("volume:anomaly (amplifier)")
            continue

        weight = SIGNAL_WEIGHTS.get(key)
        if weight is None:
            continue  # unknown signal type/value — ignore rather than guess

        confidence = sig.get("confidence_score")
        contribution = weight * (confidence if confidence is not None else 1.0)
        directional_score += contribution
        contributions.append(
            f"{sig['signal_type']}:{sig['value']} "
            f"(weight={weight:+.2f}, confidence={confidence}, contributes={contribution:+.2f})"
        )

    if has_volume_anomaly:
        directional_score *= VOLUME_ANOMALY_MULTIPLIER

    return directional_score, contributions


def _buy_side_decay(now: datetime) -> float:
    """
    1.0 during the normal trading window, linearly ramping to 0.0 exactly
    at ENTRY_CUTOFF_TIME, and 0.0 entirely outside 9:15-15:00 IST. Only
    ever applied to positive (BUY-leaning) scores — the ability to exit
    is never decayed.
    """
    now_time = now.time()
    if not (MARKET_OPEN_TIME <= now_time < ENTRY_CUTOFF_TIME):
        return 0.0
    if now_time < DECAY_START_TIME:
        return 1.0

    decay_start = datetime.combine(now.date(), DECAY_START_TIME, tzinfo=now.tzinfo)
    entry_cutoff = datetime.combine(now.date(), ENTRY_CUTOFF_TIME, tzinfo=now.tzinfo)
    window_seconds = (entry_cutoff - decay_start).total_seconds()
    elapsed_seconds = (now - decay_start).total_seconds()
    return max(0.0, 1.0 - (elapsed_seconds / window_seconds))


def _is_must_flatten(now: datetime) -> bool:
    """
    True whenever `now` falls outside the 9:15-15:15 IST safe window — the
    only time it's acceptable to be holding a position. Correct whether
    checked before market open, during the day, after close, or the next
    morning (a held position that survived the prior day's close must
    still be flagged for flattening).
    """
    now_time = now.time()
    return not (MARKET_OPEN_TIME <= now_time < FORCE_FLATTEN_TIME)


def decide(symbol: str, signals: list[dict], now: datetime | None = None) -> Decision:
    """
    Turn a symbol's recent signals into a Decision. `now` defaults to the
    current time; pass it explicitly (timezone-aware) for testing/backtesting.
    """
    now = (now or datetime.now(timezone.utc)).astimezone(IST)

    raw_score, contributions = compute_raw_score(signals)

    if raw_score > 0:
        decayed_score = raw_score * _buy_side_decay(now)
    else:
        decayed_score = raw_score  # SELL/HOLD side is never decayed

    if decayed_score >= BUY_THRESHOLD:
        action = "BUY"
    elif decayed_score <= SELL_THRESHOLD:
        action = "SELL"
    else:
        action = "HOLD"

    return Decision(
        symbol=symbol,
        raw_score=round(raw_score, 3),
        decayed_score=round(decayed_score, 3),
        action=action,
        must_flatten=_is_must_flatten(now),
        contributing_signals=contributions,
    )


async def main():
    pool = await get_db_pool()
    try:
        for symbol in settings.WATCHLIST:
            signals = await get_recent_signals(pool, symbol, minutes=LOOKBACK_MINUTES)
            decision = decide(symbol, signals)
            print(
                f"{decision.symbol}: {decision.action} "
                f"(raw={decision.raw_score}, decayed={decision.decayed_score}, "
                f"must_flatten={decision.must_flatten})"
            )
            for c in decision.contributing_signals:
                print(f"    - {c}")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
