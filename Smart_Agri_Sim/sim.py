import numpy as np
import pandas as pd

# Parameters for simulation
days = 90
np.random.seed(42)  # For reproducibility

# Soil Sensor Data
soil_moisture = np.clip(np.random.normal(loc=30, scale=5, size=days), 10, 50)  # % moisture
soil_temp = np.clip(np.random.normal(loc=25, scale=3, size=days), 15, 35)  # °C temperature
soil_npk = np.clip(np.random.normal(loc=200, scale=30, size=days), 150, 250)  # mg/kg

# Weather Data
temp = np.clip(np.random.normal(loc=28, scale=4, size=days), 20, 40)  # °C
humidity = np.clip(np.random.normal(loc=65, scale=10, size=days), 40, 90)  # %
rainfall = np.random.poisson(lam=5, size=days)  # mm/day

# Drone Imagery Data (NDVI - Normalized Difference Vegetation Index)
ndvi = np.clip(np.random.normal(loc=0.6, scale=0.1, size=days), 0.4, 0.8)  # Simulated NDVI values

# Create a DataFrame
data = pd.DataFrame({
    'Day': pd.date_range(start='2025-01-01', periods=days, freq='D'),
    'Soil Moisture (%)': soil_moisture,
    'Soil Temperature (°C)': soil_temp,
    'Soil NPK (mg/kg)': soil_npk,
    'Air Temperature (°C)': temp,
    'Humidity (%)': humidity,
    'Rainfall (mm)': rainfall,
    'NDVI': ndvi
})

# Save to CSV for future use
data.to_csv('simulated_farm_data.csv', index=False)

# Display a sample
print(data.head())

