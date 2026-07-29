"""
tests/test_decision_engine.py — Unit tests for the rule-based decision layer.

Weight values below match the recalibrated SIGNAL_WEIGHTS in
decision/decision_engine.py (sentiment:positive and pattern:support are
bearish, pattern:breakout is the only bullish signal) — see that module's
comments for the evaluation data behind these numbers.

Run with:  python -m pytest tests/ -v
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from decision.decision_engine import compute_raw_score, decide

IST = ZoneInfo("Asia/Kolkata")


def ist(hour, minute):
    return datetime(2026, 7, 21, hour, minute, tzinfo=IST)  # a Tuesday


# ── compute_raw_score ─────────────────────────────────────────────────────

def test_sentiment_positive_scores_bearish():
    # Recalibrated: positive sentiment is a bearish signal in the real data.
    signals = [{"time": ist(11, 0), "signal_type": "sentiment", "value": "positive", "confidence_score": 0.85}]
    score, contributions = compute_raw_score(signals)
    assert score == pytest.approx(-1.5 * 0.85)
    assert len(contributions) == 1


def test_pattern_breakout_alone_is_the_only_bullish_signal():
    signals = [{"time": ist(11, 0), "signal_type": "pattern", "value": "breakout", "confidence_score": None}]
    score, _ = compute_raw_score(signals)
    assert score == pytest.approx(1.0)


def test_volume_anomaly_amplifies_existing_directional_score():
    signals = [
        {"time": ist(11, 0), "signal_type": "sentiment", "value": "negative", "confidence_score": 0.9},
        {"time": ist(11, 1), "signal_type": "volume", "value": "anomaly", "confidence_score": None},
    ]
    score, _ = compute_raw_score(signals)
    assert score == pytest.approx(-0.25 * 0.9 * 1.25)


def test_volume_anomaly_alone_contributes_nothing():
    signals = [{"time": ist(11, 0), "signal_type": "volume", "value": "anomaly", "confidence_score": None}]
    score, _ = compute_raw_score(signals)
    assert score == 0.0


def test_repeated_signal_only_counts_latest_occurrence():
    # Three breakout signals from repeated engine cycles should count once, not 3x.
    signals = [
        {"time": ist(11, 0), "signal_type": "pattern", "value": "breakout", "confidence_score": None},
        {"time": ist(11, 1), "signal_type": "pattern", "value": "breakout", "confidence_score": None},
        {"time": ist(11, 2), "signal_type": "pattern", "value": "breakout", "confidence_score": None},
    ]
    score, _ = compute_raw_score(signals)
    assert score == pytest.approx(1.0)


def test_unknown_signal_type_is_ignored():
    signals = [{"time": ist(11, 0), "signal_type": "mystery", "value": "whatever", "confidence_score": None}]
    score, contributions = compute_raw_score(signals)
    assert score == 0.0
    assert contributions == []


# ── decide: thresholds ────────────────────────────────────────────────────

def test_decide_buy_requires_breakout_plus_volume_since_breakout_alone_is_below_threshold():
    # pattern:breakout (1.0) alone no longer crosses BUY_THRESHOLD (1.2) —
    # it's the only bullish-weighted signal now, so BUY realistically
    # requires the volume amplifier too: 1.0 * 1.25 = 1.25.
    signals = [
        {"time": ist(11, 0), "signal_type": "pattern", "value": "breakout", "confidence_score": None},
        {"time": ist(11, 1), "signal_type": "volume", "value": "anomaly", "confidence_score": None},
    ]
    decision = decide("RELIANCE", signals, now=ist(11, 0))
    assert decision.action == "BUY"


def test_decide_hold_when_score_below_threshold():
    signals = [{"time": ist(11, 0), "signal_type": "sentiment", "value": "positive", "confidence_score": 0.5}]
    decision = decide("RELIANCE", signals, now=ist(11, 0))
    assert decision.action == "HOLD"


def test_decide_sell_when_score_below_negative_threshold():
    # sentiment:positive at high confidence is now strong enough to trigger
    # SELL on its own: -1.5 * 1.0 = -1.5.
    signals = [{"time": ist(11, 0), "signal_type": "sentiment", "value": "positive", "confidence_score": 1.0}]
    decision = decide("RELIANCE", signals, now=ist(11, 0))
    assert decision.action == "SELL"


# ── decide: day-trading time rules ────────────────────────────────────────

def test_buy_signal_decays_within_decay_window():
    signals = [
        {"time": ist(14, 30), "signal_type": "pattern", "value": "breakout", "confidence_score": None},
        {"time": ist(14, 30), "signal_type": "volume", "value": "anomaly", "confidence_score": None},
    ]
    early = decide("RELIANCE", signals, now=ist(11, 0))   # well before decay starts, full weight
    late = decide("RELIANCE", signals, now=ist(14, 55))   # near cutoff, near-zero weight
    assert early.decayed_score > late.decayed_score
    assert early.action == "BUY"
    assert late.action == "HOLD"


def test_buy_blocked_entirely_at_or_after_entry_cutoff():
    signals = [
        {"time": ist(15, 5), "signal_type": "pattern", "value": "breakout", "confidence_score": None},
        {"time": ist(15, 5), "signal_type": "volume", "value": "anomaly", "confidence_score": None},
    ]
    decision = decide("RELIANCE", signals, now=ist(15, 5))
    assert decision.decayed_score == 0.0
    assert decision.action == "HOLD"


def test_buy_blocked_before_market_open():
    signals = [
        {"time": ist(8, 0), "signal_type": "pattern", "value": "breakout", "confidence_score": None},
        {"time": ist(8, 0), "signal_type": "volume", "value": "anomaly", "confidence_score": None},
    ]
    decision = decide("RELIANCE", signals, now=ist(8, 0))
    assert decision.decayed_score == 0.0


def test_sell_is_never_decayed_near_close():
    signals = [
        {"time": ist(15, 10), "signal_type": "sentiment", "value": "positive", "confidence_score": 1.0},
        {"time": ist(15, 10), "signal_type": "volume", "value": "anomaly", "confidence_score": None},
    ]
    decision = decide("RELIANCE", signals, now=ist(15, 10))
    assert decision.action == "SELL"
    assert decision.decayed_score == decision.raw_score


def test_must_flatten_false_during_safe_window():
    decision = decide("RELIANCE", [], now=ist(12, 0))
    assert decision.must_flatten is False


def test_must_flatten_true_at_force_flatten_time():
    decision = decide("RELIANCE", [], now=ist(15, 15))
    assert decision.must_flatten is True


def test_must_flatten_true_after_market_close():
    decision = decide("RELIANCE", [], now=ist(20, 0))
    assert decision.must_flatten is True


def test_must_flatten_true_before_market_open_next_day():
    decision = decide("RELIANCE", [], now=ist(8, 0))
    assert decision.must_flatten is True
