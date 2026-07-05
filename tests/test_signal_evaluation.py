"""
tests/test_signal_evaluation.py — Unit tests for signal evaluation.

Run with:  python -m pytest tests/ -v
"""

import pytest

from evaluation.signal_evaluation import compute_forward_returns, summarize_by_signal


def test_directional_signal_win_rate():
    rows = [
        {"symbol": "RELIANCE", "signal_type": "sentiment", "value": "positive",
         "confidence_score": 0.9, "entry_price": 100.0, "exit_price": 102.0, "horizon_minutes": 15},
        {"symbol": "TCS", "signal_type": "sentiment", "value": "positive",
         "confidence_score": 0.8, "entry_price": 100.0, "exit_price": 98.0, "horizon_minutes": 15},
    ]
    df = compute_forward_returns(rows)
    summary = summarize_by_signal(df)

    row = summary.iloc[0]
    assert row["signal_type"] == "sentiment"
    assert row["value"] == "positive"
    assert row["count"] == 2
    assert row["win_rate_pct"] == 50.0  # one up, one down


def test_negative_sentiment_direction_is_flipped():
    rows = [
        {"symbol": "RELIANCE", "signal_type": "sentiment", "value": "negative",
         "confidence_score": 0.9, "entry_price": 100.0, "exit_price": 95.0, "horizon_minutes": 15},
    ]
    df = compute_forward_returns(rows)
    summary = summarize_by_signal(df)

    row = summary.iloc[0]
    # Price fell, which is what a negative-sentiment signal predicts -> a "win"
    assert row["win_rate_pct"] == 100.0
    assert row["mean_return_pct"] > 0  # directional return is positive (correct call)


def test_volume_signal_has_no_win_rate_only_magnitude():
    rows = [
        {"symbol": "RELIANCE", "signal_type": "volume", "value": "anomaly",
         "confidence_score": None, "entry_price": 100.0, "exit_price": 103.0, "horizon_minutes": 5},
    ]
    df = compute_forward_returns(rows)
    summary = summarize_by_signal(df)

    row = summary.iloc[0]
    assert row["win_rate_pct"] is None
    assert row["mean_return_pct"] == pytest.approx(3.0)


def test_summarize_empty_input_returns_empty_dataframe():
    df = compute_forward_returns([])
    summary = summarize_by_signal(df)
    assert summary.empty
