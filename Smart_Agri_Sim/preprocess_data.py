from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load the dataset
data = pd.read_csv('simulated_farm_data.csv')

# Check for missing values
if data.isnull().sum().sum() > 0:
    print("Missing values detected. Filling missing values...")
    data = data.fillna(method='ffill')  # Forward fill missing values

# Normalize the numerical features for LSTM
scaler = MinMaxScaler()
scaled_data = data.copy()
scaled_columns = ['Soil Moisture (%)', 'Soil Temperature (°C)', 'Soil NPK (mg/kg)',
                  'Air Temperature (°C)', 'Humidity (%)', 'Rainfall (mm)', 'NDVI']

scaled_data[scaled_columns] = scaler.fit_transform(data[scaled_columns])

# Split data into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)

# Save the scaler for inverse-transforming predictions later
joblib.dump(scaler, 'scaler.pkl')

# Display the first few rows of the pre-processed data
print("Pre-processed data (scaled):")
print(scaled_data.head())

# Save processed data
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# Optional: Save a snapshot to visually check the data
print("Train and test data saved as 'train_data.csv' and 'test_data.csv'")
