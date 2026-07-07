"""
evaluation/signal_evaluation.py — Checks whether signals from the signal
engine actually predict favorable forward price moves.

For each signal, looks up an entry price (last tick at/before the signal)
and an exit price (first tick at/after signal_time + horizon), then
aggregates the forward return by (signal_type, value, horizon) so we can
see whether any signal type has real predictive edge before building a
decision layer on top of them.

Run with:  python -m evaluation.signal_evaluation
"""

import asyncio
import sys

import pandas as pd
from loguru import logger

from storage.timescale import get_db_pool, get_signal_forward_prices

# Windows consoles default to a codepage (e.g. cp1252) that can't encode the
# em dashes used in log/print messages below — force UTF-8 on both streams
# (loguru's default handler writes to stderr, not stdout) so they render
# correctly instead of as replacement characters.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_HORIZONS_MINUTES = (5, 15, 60)

# Expected price direction for each (signal_type, value) pair.
# +1 = signal predicts price up, -1 = predicts price down,
# None = no directional prediction (evaluated on return magnitude only).
EXPECTED_DIRECTION = {
    ("sentiment", "positive"): 1,
    ("sentiment", "negative"): -1,
    ("pattern", "breakout"): 1,
    ("pattern", "support"): 1,
    ("volume", "anomaly"): None,
}


def compute_forward_returns(rows: list[dict]) -> pd.DataFrame:
    """Turn raw (signal, entry_price, exit_price) rows into a returns DataFrame."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["forward_return"] = (df["exit_price"] - df["entry_price"]) / df["entry_price"]
    df["expected_direction"] = df.apply(
        lambda row: EXPECTED_DIRECTION.get((row["signal_type"], row["value"])), axis=1
    )
    df["expected_direction"] = pd.to_numeric(df["expected_direction"], errors="coerce")
    df["directional_return"] = df["forward_return"] * df["expected_direction"].fillna(0)
    return df


def summarize_by_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate forward returns per (signal_type, value, horizon_minutes)."""
    columns = [
        "signal_type", "value", "horizon_minutes", "count",
        "mean_return_pct", "median_return_pct", "win_rate_pct",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (signal_type, value, horizon), group in df.groupby(
        ["signal_type", "value", "horizon_minutes"]
    ):
        has_direction = group["expected_direction"].notna().all()
        if has_direction:
            win_rate_pct = 100 * (group["directional_return"] > 0).sum() / len(group)
            mean_return_pct = 100 * group["directional_return"].mean()
            median_return_pct = 100 * group["directional_return"].median()
        else:
            # No directional prediction (e.g. volume anomaly) — report raw magnitude instead.
            win_rate_pct = None
            mean_return_pct = 100 * group["forward_return"].mean()
            median_return_pct = 100 * group["forward_return"].median()

        rows.append({
            "signal_type": signal_type,
            "value": value,
            "horizon_minutes": horizon,
            "count": len(group),
            "mean_return_pct": round(mean_return_pct, 3),
            "median_return_pct": round(median_return_pct, 3),
            "win_rate_pct": round(win_rate_pct, 1) if win_rate_pct is not None else None,
        })

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["signal_type", "value", "horizon_minutes"]
    )


async def run_evaluation(pool, horizons_minutes=DEFAULT_HORIZONS_MINUTES) -> pd.DataFrame:
    """Fetch signals + forward prices for each horizon and summarize the results."""
    all_rows = []
    for horizon in horizons_minutes:
        rows = await get_signal_forward_prices(pool, horizon)
        for row in rows:
            row["horizon_minutes"] = horizon
        all_rows.extend(rows)

    if not all_rows:
        logger.warning("No signals with matching forward prices found — nothing to evaluate.")
        return pd.DataFrame()

    df = compute_forward_returns(all_rows)
    return summarize_by_signal(df)


async def main():
    pool = await get_db_pool()
    try:
        summary = await run_evaluation(pool)
        if summary.empty:
            print("No signals found yet — run the bot for a while and try again.")
        else:
            print(summary.to_string(index=False))
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
