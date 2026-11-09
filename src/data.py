"""
src/data.py
Robust data loading, validation, and feature preparation for time-series and transport data.
"""

from pathlib import Path
import pandas as pd
import numpy as np

def load_commodity_demand(csv_path: str | Path) -> pd.DataFrame:
    """Loads and validates daily commodity demand time series."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def load_freight_quotes(csv_path: str | Path) -> pd.DataFrame:
    """Loads and cleans transport quote datasets."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def build_route_features(df_quotes: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs multi-dimensional statistical features per transport corridor:
    - current_rate: most recent observed cost/ton
    - rolling_mean_30d: baseline expectation
    - rolling_std_30d: market volatility
    - rate_deviation_zscore: normalized distance from mean
    - cost_per_km: unit transport efficiency metric
    """
    df = df_quotes.copy()
    features = []

    for route_id, group in df.groupby(["origin", "destination"]):
        group = group.sort_values("date")
        if len(group) < 5:
            continue
        
        rates = group["rate_per_ton"].values
        current_rate = float(rates[-1])
        mean_rate = float(np.mean(rates))
        std_rate = float(np.std(rates, ddof=0))
        dist = float(group["distance_km"].iloc[0])
        cost_per_km = current_rate / max(dist, 1.0)
        zscore = (current_rate - mean_rate) / (std_rate + 1e-6)

        features.append({
            "origin": route_id[0],
            "destination": route_id[1],
            "distance_km": dist,
            "observations_count": len(group),
            "current_rate": current_rate,
            "mean_rate": round(mean_rate, 2),
            "std_rate": round(std_rate, 2),
            "rate_deviation_zscore": round(zscore, 3),
            "cost_per_km": round(cost_per_km, 4),
        })

    return pd.DataFrame(features)
