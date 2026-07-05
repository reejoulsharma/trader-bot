import pandas as pd

class VolumeAnomalyDetector:
    def __init__(self, window_size=20, spike_threshold=2.0):
        """
        :param window_size: Number of periods for moving average
        :param spike_threshold: Multiplier for anomaly detection (e.g. 2.0 = 200% of MA)
        """
        self.window_size = window_size
        self.spike_threshold = spike_threshold

    def analyze(self, df: pd.DataFrame) -> bool:
        """
        Analyzes a dataframe of recent ticks or candles.
        Expected columns must include 'volume'.
        Returns True if the latest volume is an anomaly.
        """
        if df.empty or len(df) < self.window_size:
            return False
            
        if 'volume' not in df.columns:
            return False
            
        # Calculate moving average of volume
        vol_ma = df['volume'].rolling(window=self.window_size).mean()
        
        # Get the latest values
        latest_vol = df['volume'].iloc[-1]
        latest_ma = vol_ma.iloc[-1]
        
        if pd.isna(latest_ma) or latest_ma <= 0:
            return False
            
        if latest_vol > (latest_ma * self.spike_threshold):
            return True
            
        return False
