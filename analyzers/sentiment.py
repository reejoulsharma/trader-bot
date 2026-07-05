import torch
from transformers import pipeline

class FinBERTSentimentAnalyzer:
    def __init__(self):
        # Determine device (0 for GPU, -1 for CPU)
        self.device = 0 if torch.cuda.is_available() else -1
        # Load the pipeline for financial sentiment analysis
        self.nlp = pipeline(
            "sentiment-analysis", 
            model="ProsusAI/finbert", 
            device=self.device
        )

    def analyze(self, text: str) -> dict:
        """
        Returns a dict with 'label' (positive/negative/neutral) and 'score'
        """
        if not text:
            return {"label": "neutral", "score": 0.0}
        
        try:
            # We truncate to 512 tokens to avoid errors with long texts
            results = self.nlp(text, truncation=True, max_length=512)
            if results:
                return results[0]
        except Exception as e:
            # Fallback if inference fails
            pass
            
        return {"label": "neutral", "score": 0.0}
