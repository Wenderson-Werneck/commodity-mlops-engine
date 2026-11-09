"""
src/anomaly.py
Unsupervised anomaly detection for logistics rates and operational pricing
using IsolationForest with domain-specific statistical feature engineering.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class FreightAnomalyDetector:
    """
    Detects irregular transport quotes across industrial corridors.
    Separates structural differences (e.g. longer distance routes) from true statistical outliers.
    """

    FEATURE_COLS = ["distance_km", "rate_per_ton", "cargo_tonnage", "diesel_price_ref"]

    def __init__(self, contamination: float = 0.05, random_state: int = 42, n_estimators: int = 150):
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            n_jobs=-1
        )
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "FreightAnomalyDetector":
        X = df[self.FEATURE_COLS].values
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Appends anomaly prediction flag (-1: anomaly, 1: normal) and continuous anomaly score.
        Lower score indicates higher likelihood of anomaly.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict.")

        df_out = df.copy()
        X = df_out[self.FEATURE_COLS].values
        
        preds = self.model.predict(X)
        scores = self.model.decision_function(X)

        df_out["anomaly_label"] = np.where(preds == -1, "ANOMALY", "NORMAL")
        df_out["is_anomaly"] = np.where(preds == -1, 1, 0)
        df_out["anomaly_score"] = np.round(scores, 4)

        return df_out

    def evaluate_synthetic_ground_truth(self, df_scored: pd.DataFrame) -> Dict[str, float]:
        """Calculates precision, recall, and detection rate against injected synthetic flags."""
        if "is_synthetic_anomaly" not in df_scored.columns:
            return {}

        y_true = df_scored["is_synthetic_anomaly"].values
        y_pred = df_scored["is_anomaly"].values

        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))

        precision = true_positives / (true_positives + false_positives + 1e-9)
        recall = true_positives / (true_positives + false_negatives + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        return {
            "precision": float(round(precision, 4)),
            "recall": float(round(recall, 4)),
            "f1_score": float(round(f1, 4)),
            "detected_anomalies_count": int(np.sum(y_pred == 1)),
        }
