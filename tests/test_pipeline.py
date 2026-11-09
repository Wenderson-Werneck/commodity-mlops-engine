"""
tests/test_pipeline.py
Unit tests verifying data generation, feature engineering,
anomaly detection, and statistical forecasting.
"""

import pytest
import numpy as np
import pandas as pd

from src.anomaly import FreightAnomalyDetector
from src.forecast import CommodityDemandForecaster
from src.data import build_route_features

@pytest.fixture
def mock_freight_data():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "origin": ["Port A"] * 50 + ["Port B"] * 50,
        "destination": ["Plant X"] * 50 + ["Plant Y"] * 50,
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "distance_km": [350.0] * 50 + [600.0] * 50,
        "cargo_tonnage": np.random.uniform(30.0, 45.0, n),
        "diesel_price_ref": np.random.uniform(5.5, 6.2, n),
        "rate_per_ton": np.random.normal(180.0, 15.0, n),
        "is_synthetic_anomaly": [0] * 95 + [1] * 5
    })

@pytest.fixture
def mock_demand_series():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    values = 100 + 0.1 * np.arange(180) + 10 * np.sin(2 * np.pi * np.arange(180) / 7) + np.random.normal(0, 2, 180)
    return pd.Series(values, index=dates)

def test_freight_anomaly_detector_fit_predict(mock_freight_data):
    detector = FreightAnomalyDetector(contamination=0.05, random_state=42)
    detector.fit(mock_freight_data)
    assert detector.is_fitted

    scored = detector.predict(mock_freight_data)
    assert "anomaly_label" in scored.columns
    assert "anomaly_score" in scored.columns
    assert set(scored["anomaly_label"].unique()).issubset({"NORMAL", "ANOMALY"})

def test_commodity_forecaster_fit_and_forecast(mock_demand_series):
    forecaster = CommodityDemandForecaster(seasonal_periods=7, holdout_days=20)
    forecaster.fit(mock_demand_series)

    assert forecaster.best_model is not None
    assert "val_wape" in forecaster.metrics
    assert forecaster.metrics["val_wape"] < 0.15 # Under 15% error on synthetic data

    forecast_14d = forecaster.predict_future(steps=14)
    assert len(forecast_14d) == 14
    assert np.all(forecast_14d > 0)

def test_route_features_calculation(mock_freight_data):
    feats = build_route_features(mock_freight_data)
    assert len(feats) == 2
    assert "rate_deviation_zscore" in feats.columns
    assert "cost_per_km" in feats.columns
