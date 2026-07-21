import pandas as pd

class PricePatternAnalyzer:
    def __init__(self):
        pass

    def detect_gap_up(self, previous_close: float, current_open: float, threshold_pct: float = 0.01) -> bool:
        """
        Detects a gap up between previous close and current open.
        """
        if previous_close <= 0:
            return False
        gap = (current_open - previous_close) / previous_close
        return gap >= threshold_pct

    def _past_window_prices(self, df: pd.DataFrame, price_col: str, window_minutes: int) -> pd.Series | None:
        """
        Prices strictly before the latest tick, within the last `window_minutes`
        of the latest tick's timestamp. Time-based rather than a fixed row
        count, since tick arrival rate varies a lot (a count-based window can
        collapse to a few seconds of near-flat price during quiet periods,
        making support/breakout trivially true on noise rather than a real
        multi-minute pattern).
        """
        if price_col not in df.columns or 'timestamp' not in df.columns or df.empty or len(df) < 2:
            return None

        # format="ISO8601" handles per-row precision differences (Python's
        # isoformat() omits microseconds when they're exactly zero, so a
        # mix of timestamps with/without fractional seconds is expected,
        # not an edge case — pandas' fast-path parser otherwise infers one
        # fixed format from the first rows and raises on the rest).
        timestamps = pd.to_datetime(df['timestamp'], utc=True, format="ISO8601")
        latest_time = timestamps.iloc[-1]
        window_start = latest_time - pd.Timedelta(minutes=window_minutes)

        past_mask = (timestamps >= window_start) & (timestamps < latest_time)
        past_window = df.loc[past_mask, price_col]
        return past_window if not past_window.empty else None

    def detect_breakout(self, df: pd.DataFrame, window_minutes: int = 15) -> bool:
        """
        Checks if the latest price is higher than the max price over the
        previous `window_minutes` of history (excluding the current tick).
        Expected columns: ['price'] or ['close'], and 'timestamp'.
        """
        price_col = 'close' if 'close' in df.columns else 'price'
        past_window = self._past_window_prices(df, price_col, window_minutes)
        if past_window is None:
            return False

        highest = past_window.max()
        latest_price = df[price_col].iloc[-1]
        return bool(latest_price > highest)

    def detect_support(self, df: pd.DataFrame, window_minutes: int = 10, tolerance_pct: float = 0.005) -> bool:
        """
        Checks if the current price is near the local minimum (support)
        over the previous `window_minutes` of history.
        """
        price_col = 'close' if 'close' in df.columns else 'price'
        past_window = self._past_window_prices(df, price_col, window_minutes)
        if past_window is None:
            return False

        lowest = past_window.min()
        latest_price = df[price_col].iloc[-1]

        # Is the price near the recent support level?
        return bool(lowest * (1 - tolerance_pct) <= latest_price <= lowest * (1 + tolerance_pct))
