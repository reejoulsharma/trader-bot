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

    def detect_breakout(self, df: pd.DataFrame, window: int = 20) -> bool:
        """
        Checks if the latest price is higher than the max of the previous `window` periods.
        Expected columns: ['price'] or ['close']
        """
        price_col = 'close' if 'close' in df.columns else 'price'
        if price_col not in df.columns or df.empty or len(df) < window + 1:
            return False
            
        # Previous window max (excluding the current/latest tick)
        past_window = df[price_col].iloc[-(window+1):-1]
        highest = past_window.max()
        
        latest_price = df[price_col].iloc[-1]
        
        return latest_price > highest

    def detect_support(self, df: pd.DataFrame, window: int = 14, tolerance_pct: float = 0.005) -> bool:
        """
        Checks if the current price is near a local minimum (support) over the last `window` periods.
        """
        price_col = 'close' if 'close' in df.columns else 'price'
        if price_col not in df.columns or df.empty or len(df) < window + 1:
            return False
            
        past_window = df[price_col].iloc[-(window+1):-1]
        lowest = past_window.min()
        
        latest_price = df[price_col].iloc[-1]
        
        # Is the price near the recent support level?
        if lowest * (1 - tolerance_pct) <= latest_price <= lowest * (1 + tolerance_pct):
            return True
        return False
