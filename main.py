"""
main.py
End-to-End Orchestrator for the Commodity & Freight MLOps Suite.
Executes data loading, anomaly detection, time-series forecasting,
and logs parameters and metrics to MLflow.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from scripts.generate_data import generate_synthetic_datasets
from src.data import load_commodity_demand, load_freight_quotes, build_route_features
from src.anomaly import FreightAnomalyDetector
from src.forecast import CommodityDemandForecaster
from src.tracker import MLOpsExperimentTracker

def run_pipeline() -> None:
    load_dotenv()
    print("=" * 70)
    print("  COMMODITY MLOps ENGINE - PRODUCTION REFERENCE BENCHMARK")
    print("=" * 70)

    data_dir = Path("data")
    demand_file = data_dir / "synthetic_commodity_demand.csv"
    freight_file = data_dir / "synthetic_freight_quotes.csv"

    # Step 0: Ensure synthetic data exists
    if not demand_file.exists() or not freight_file.exists():
        print("\n[Step 0] Synthetic datasets missing. Generating benchmark data...")
        generate_synthetic_datasets(output_dir=str(data_dir))
    else:
        print("\n[Step 0] Validated synthetic datasets in ./data")

    # Step 1: Initialize MLflow Tracker
    tracker = MLOpsExperimentTracker(experiment_name="commodity_logistics_mlops")

    # Step 2: Time-Series Demand Forecasting Pipeline
    print("\n[Step 1/2] Training Commodity Demand Forecaster (Holt-Winters + Damped Trend)...")
    df_demand = load_commodity_demand(demand_file)
    series = df_demand.set_index("date")["consumption_tons"]

    forecaster = CommodityDemandForecaster(seasonal_periods=7, holdout_days=30)
    forecaster.fit(series)

    horizon_14d = forecaster.predict_future(steps=14)
    print(f" -> Optimal Holt-Winters Config: {forecaster.best_config}")
    print(f" -> Validation WAPE: {forecaster.metrics['val_wape']:.2%}")
    print(f" -> Validation RMSE: {forecaster.metrics['val_rmse']:.2f} tons")
    print(f" -> 14-Day Forward Forecast Sample: {horizon_14d[:5]}...")

    # Log to MLflow
    tracker.log_run(
        run_name="holt_winters_consumption_forecast",
        params={"seasonal_periods": 7, "holdout_days": 30, **{f"kwarg_{k}": str(v) for k, v in forecaster.best_config.items()}},
        metrics=forecaster.metrics,
        model_object=forecaster.best_model,
        artifact_save_path="artifacts/demand_forecaster.joblib"
    )

    # Step 3: Unsupervised Freight Anomaly Detection
    print("\n[Step 2/2] Training Freight Anomaly Detector (IsolationForest)...")
    df_freight = load_freight_quotes(freight_file)

    detector = FreightAnomalyDetector(contamination=0.04, random_state=42, n_estimators=150)
    detector.fit(df_freight)
    df_scored = detector.predict(df_freight)

    eval_metrics = detector.evaluate_synthetic_ground_truth(df_scored)
    print(f" -> Analyzed {len(df_scored)} quotes across all corridors")
    print(f" -> Injected Ground Truth Detected:")
    print(f"    - Precision: {eval_metrics.get('precision', 0.0):.2%}")
    print(f"    - Recall:    {eval_metrics.get('recall', 0.0):.2%}")
    print(f"    - F1 Score:  {eval_metrics.get('f1_score', 0.0):.2f}")

    # Log to MLflow
    tracker.log_run(
        run_name="isolation_forest_freight_anomaly",
        params={"contamination": 0.04, "n_estimators": 150, "random_state": 42},
        metrics=eval_metrics,
        model_object=detector.model,
        artifact_save_path="artifacts/freight_anomaly_detector.joblib"
    )

    print("\n" + "=" * 70)
    print("  [SUCCESS] All pipeline stages executed & logged to MLflow.")
    print("  To inspect the MLflow dashboard locally:")
    print("  $ mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
