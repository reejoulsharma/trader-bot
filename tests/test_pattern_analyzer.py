"""
tests/test_pattern_analyzer.py — Unit tests for PricePatternAnalyzer.

Run with:  python -m pytest tests/ -v
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from analyzers.pattern import PricePatternAnalyzer

analyzer = PricePatternAnalyzer()


def ticks(prices_with_offsets_seconds: list[tuple[float, int]]) -> pd.DataFrame:
    """Build a tick DataFrame from (price, seconds_ago) pairs, oldest first."""
    base = datetime(2026, 7, 21, 10, 0, 0, tzinfo=timezone.utc)
    rows = [
        {"timestamp": (base - timedelta(seconds=offset)).isoformat(), "price": price, "volume": 0}
        for price, offset in prices_with_offsets_seconds
    ]
    rows.sort(key=lambda r: r["timestamp"])
    return pd.DataFrame(rows)


# ── detect_breakout ──────────────────────────────────────────────────────

def test_breakout_true_when_latest_price_exceeds_recent_max():
    df = ticks([
        (100.0, 600), (100.5, 500), (101.0, 400), (100.8, 300),
        (100.2, 5), (102.0, 0),  # latest tick breaks above all of the above
    ])
    assert analyzer.detect_breakout(df, window_minutes=15) is True


def test_breakout_false_when_latest_price_within_recent_range():
    df = ticks([
        (100.0, 600), (101.0, 500), (100.5, 400), (100.8, 300), (100.9, 0),
    ])
    assert analyzer.detect_breakout(df, window_minutes=15) is False


def test_breakout_ignores_ticks_outside_the_time_window():
    # A high price 20 minutes ago (outside a 15-minute window) shouldn't
    # block a breakout above the genuinely recent range.
    df = ticks([
        (150.0, 20 * 60),   # outside window — would have blocked a naive row-count check
        (100.0, 600), (100.2, 300), (100.1, 60),
        (100.5, 0),          # breaks above the in-window range, not the old spike
    ])
    assert analyzer.detect_breakout(df, window_minutes=15) is True


# ── detect_support ────────────────────────────────────────────────────────

def test_support_true_when_latest_price_near_recent_minimum():
    df = ticks([
        (100.0, 600), (99.5, 400), (99.4, 200), (99.42, 0),  # within 0.5% of 99.4 low
    ])
    assert analyzer.detect_support(df, window_minutes=10, tolerance_pct=0.005) is True


def test_support_false_when_latest_price_far_from_minimum():
    df = ticks([
        (100.0, 600), (99.5, 400), (99.4, 200), (105.0, 0),
    ])
    assert analyzer.detect_support(df, window_minutes=10, tolerance_pct=0.005) is False


def test_support_uses_real_window_low_not_just_recent_noise():
    # This is the actual bug this fix addresses. A genuine dip to 95 five
    # minutes ago is the real support level. But a burst of high-frequency
    # ticks in the last few seconds — all clustered near 100, pure noise —
    # would, under the old row-count window (e.g. last 14 rows), push that
    # real dip out of view entirely: the analyzer would only ever see the
    # last few seconds of near-flat noise and call the current price
    # "at support" against that noise floor, a false positive. The
    # time-based window must still see the real 95 low and correctly
    # recognise that 100 is nowhere near it.
    noise = [(100.0 + i * 0.01, 10 - i * 0.5) for i in range(20)]  # last ~10s, clustered ~100
    df = ticks([(100.0, 300), (95.0, 300 - 1)] + noise)
    assert analyzer.detect_support(df, window_minutes=10, tolerance_pct=0.005) is False


# ── shared edge cases ────────────────────────────────────────────────────

def test_empty_dataframe_returns_false():
    df = pd.DataFrame(columns=["timestamp", "price"])
    assert analyzer.detect_breakout(df) is False
    assert analyzer.detect_support(df) is False


def test_missing_timestamp_column_returns_false():
    df = pd.DataFrame([{"price": 100.0}, {"price": 101.0}])
    assert analyzer.detect_breakout(df) is False
    assert analyzer.detect_support(df) is False


def test_single_row_returns_false():
    df = ticks([(100.0, 0)])
    assert analyzer.detect_breakout(df) is False
    assert analyzer.detect_support(df) is False
