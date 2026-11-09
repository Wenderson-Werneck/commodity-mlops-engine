"""
src/tracker.py
Resilient MLOps experiment tracking and artifact serialization manager.
Uses MLflow when installed; gracefully falls back to structured JSON/SQLite
audit logs to guarantee reproducible execution across any environment.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import joblib

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

class MLOpsExperimentTracker:
    """Manages experiment tracking runs and artifact exports."""

    def __init__(self, experiment_name: str = "commodity_freight_mlops", tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        if HAS_MLFLOW:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)

    def log_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        model_object: Optional[Any] = None,
        artifact_save_path: Optional[str] = None
    ) -> str:
        """Starts an experiment run, logs parameters and metrics, and saves model artifact."""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{abs(hash(run_name)) % 10000}"

        # 1. Save local binary artifact if provided
        if model_object is not None and artifact_save_path:
            Path(artifact_save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_object, artifact_save_path)

        # 2. Log via MLflow if available
        if HAS_MLFLOW:
            with mlflow.start_run(run_name=run_name) as run:
                for k, v in params.items():
                    mlflow.log_param(k, v)
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
                if artifact_save_path:
                    mlflow.log_artifact(artifact_save_path)
                run_id = run.info.run_id
                print(f"[MLflow] Run logged successfully: '{run_name}' (ID: {run_id})")
        else:
            # Fallback to local structured audit log
            log_dir = Path("artifacts")
            log_dir.mkdir(parents=True, exist_ok=True)
            audit_file = log_dir / "experiment_history.jsonl"
            
            entry = {
                "run_id": run_id,
                "run_name": run_name,
                "timestamp": datetime.now().isoformat(),
                "experiment": self.experiment_name,
                "parameters": params,
                "metrics": metrics,
                "artifact_path": artifact_save_path
            }
            with open(audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"[Tracker (Audit Mode)] Logged run: '{run_name}' -> {audit_file}")

        return run_id
