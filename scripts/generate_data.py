"""
scripts/generate_data.py
Generates 100% synthetic, realistic industrial benchmark data for:
1. Port-to-Plant Road Freight Quotes with market volatility and rare injected anomalies.
2. Daily Smelting Raw Material Consumption (Time Series with weekly patterns and trend).

Zero proprietary or client information is used.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

def generate_synthetic_datasets(output_dir: str = "data") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # 1. Daily Time Series: Commodity Consumption / Demand (24 months)
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
    n_days = len(dates)

    # Base trend + weekly seasonality + industrial cyclic noise
    trend = np.linspace(120, 185, n_days)
    weekly_seasonality = 15 * np.sin(2 * np.pi * dates.dayofweek / 7)
    noise = np.random.normal(0, 8, n_days)
    consumption = np.clip(trend + weekly_seasonality + noise, 40, None)

    df_demand = pd.DataFrame({
        "date": dates,
        "commodity_grade": "Industrial-FerroAlloy-Standard",
        "consumption_tons": np.round(consumption, 2),
        "active_kilns": np.random.choice([3, 4, 5], size=n_days, p=[0.2, 0.6, 0.2]),
        "market_price_index": np.round(2800 + np.cumsum(np.random.normal(0.5, 12, n_days)), 2)
    })

    demand_path = os.path.join(output_dir, "synthetic_commodity_demand.csv")
    df_demand.to_csv(demand_path, index=False)
    print(f"[OK] Generated {n_days} days of demand time-series -> {demand_path}")

    # 2. Freight Transport Quotes & Dispatches (5 Routes over 180 days)
    routes = [
        {"origin": "Deepwater Port Terminal Alpha", "dest": "Industrial Complex North", "dist_km": 420, "base_rate": 185.0},
        {"origin": "Deepwater Port Terminal Alpha", "dest": "Smelting Facility Beta", "dist_km": 680, "base_rate": 265.0},
        {"origin": "Coastal Bulk Hub Charlie", "dest": "Manufacturing Cluster Delta", "dist_km": 310, "base_rate": 145.0},
        {"origin": "Coastal Bulk Hub Charlie", "dest": "Refinery Zone Epsilon", "dist_km": 540, "base_rate": 220.0},
        {"origin": "Intermodal Railhead Gamma", "dest": "Smelting Facility Beta", "dist_km": 210, "base_rate": 95.0},
    ]

    records = []
    quote_id = 10001
    recent_dates = pd.date_range(start="2025-07-01", end="2025-12-31", freq="D")

    for dt in recent_dates:
        # 3 to 8 shipments per day across routes
        n_shipments = np.random.randint(3, 9)
        for _ in range(n_shipments):
            route = routes[np.random.choice(len(routes))]
            tonnage = np.random.uniform(26.0, 48.0) # Bulk tipper truck
            
            # Rate calculation: base distance cost + fuel fluctuation + random noise
            diesel_index = 5.80 + np.sin(dt.dayofyear / 20) * 0.40
            normal_rate_per_ton = route["base_rate"] + (diesel_index * 12.0) + np.random.normal(0, 6.0)
            
            # Inject ~4% rare operational anomalies (billing mistakes, emergency premiums)
            is_anomaly = np.random.rand() < 0.04
            if is_anomaly:
                # Disproportionately high or suspiciously low
                multiplier = np.random.choice([1.75, 0.45])
                actual_rate_per_ton = normal_rate_per_ton * multiplier
            else:
                actual_rate_per_ton = normal_rate_per_ton

            records.append({
                "quote_id": f"QT-{quote_id}",
                "date": dt.strftime("%Y-%m-%d"),
                "origin": route["origin"],
                "destination": route["dest"],
                "distance_km": route["dist_km"],
                "cargo_tonnage": round(tonnage, 2),
                "diesel_price_ref": round(diesel_index, 2),
                "rate_per_ton": round(actual_rate_per_ton, 2),
                "total_freight_cost": round(actual_rate_per_ton * tonnage, 2),
                "is_synthetic_anomaly": int(is_anomaly)
            })
            quote_id += 1

    df_freight = pd.DataFrame(records)
    freight_path = os.path.join(output_dir, "synthetic_freight_quotes.csv")
    df_freight.to_csv(freight_path, index=False)
    print(f"[OK] Generated {len(df_freight)} freight dispatch quotes -> {freight_path}")

if __name__ == "__main__":
    generate_synthetic_datasets()
