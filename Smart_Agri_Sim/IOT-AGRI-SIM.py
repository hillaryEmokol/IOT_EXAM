#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

def run_simulation(num_days=180, irrigation_threshold=25, test_ratio=0.2):
    """
    Run an IoT agriculture simulation for 'num_days' of synthetic data.
    'irrigation_threshold' is the soil moisture % below which we decide irrigation is needed.
    'test_ratio' defines the fraction of data used for testing (e.g., 0.2 for 20%).
    """

    # 1. Generate Synthetic Data
    np.random.seed(42)  # For reproducibility
    days = np.arange(num_days)

    # Soil moisture: ~30% average, ±5% std dev
    soil_moisture = np.random.normal(loc=30, scale=5, size=num_days)
    # Temperature: ~25°C average, ±3°C
    temperature = np.random.normal(loc=25, scale=3, size=num_days)
    # Rainfall: 60% chance of 0, 20% chance of 5mm, 15% chance of 10mm, 5% chance of 20mm
    rainfall = np.random.choice([0, 5, 10, 20], size=num_days, p=[0.6, 0.2, 0.15, 0.05])

    # Construct DataFrame
    df = pd.DataFrame({
        'day': days,
        'soil_moisture': soil_moisture,
        'temperature': temperature,
        'rainfall': rainfall
    })

    # Calculate a rolling average of soil moisture (3-day window)
    df['soil_moisture_3d_avg'] = df['soil_moisture'].rolling(window=3, min_periods=1).mean()

    # Next-day soil moisture (regression target)
    df['soil_moisture_next'] = df['soil_moisture'].shift(-1)

    # Drop last row with NaN in next-day moisture
    df = df.dropna()

    # 2. Regression Model: Predict next day's soil moisture
    features = ['soil_moisture', 'temperature', 'rainfall', 'soil_moisture_3d_avg']
    target = 'soil_moisture_next'

    X = df[features]
    y = df[target]

    # Split data (chronologically - shuffle=False to mimic time progression)
    train_size = int((1 - test_ratio) * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    reg_model = RandomForestRegressor(n_estimators=50, random_state=42)
    reg_model.fit(X_train, y_train)
    y_pred_reg = reg_model.predict(X_test)
    mae_reg = mean_absolute_error(y_test, y_pred_reg)

    # 3. Classification Model: "Irrigation Needed?" if next day's moisture < threshold
    df['irrigation_needed'] = (df['soil_moisture_next'] < irrigation_threshold).astype(int)
    y_cls = df['irrigation_needed']

    y_cls_train, y_cls_test = y_cls.iloc[:train_size], y_cls.iloc[train_size:]

    cls_model = RandomForestClassifier(n_estimators=50, random_state=42)
    cls_model.fit(X_train, y_cls_train)
    y_pred_cls = cls_model.predict(X_test)
    acc_cls = accuracy_score(y_cls_test, y_pred_cls)

    # 4. Compare Resource Usage: "Smart" vs. "Fixed"
    days_irrigated_smart = np.sum(y_pred_cls)  # # of '1's
    days_irrigated_fixed = len(X_test)  # water every day in the test set

    # Print results
    print("=== IoT Smart Agriculture Simulation ===")
    print(f"Total Days: {num_days}, Train Days: {train_size}, Test Days: {len(X_test)}\n")

    print("Sample Synthetic Data (first 5 rows):")
    print(df.head(5), "\n")

    print("Regression: Predicting Next-Day Soil Moisture")
    print(f"- Mean Absolute Error (Test): {mae_reg:.2f}%\n")

    print("Classification: Irrigation Needed?")
    print(f"- Accuracy (Test): {acc_cls*100:.2f}%\n")

    print("Resource Usage Comparison (Test Set Only)")
    print(f"- Smart Approach (predicted): irrigated {int(days_irrigated_smart)} out of {len(X_test)} days")
    print(f"- Fixed Approach: irrigated {days_irrigated_fixed} out of {len(X_test)} days")
    print("==========================================\n")

if __name__ == "__main__":
    # Run the simulation with default parameters
    run_simulation()
