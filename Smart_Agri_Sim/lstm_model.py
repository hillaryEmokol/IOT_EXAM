import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load pre-processed data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Prepare the data for LSTM
def create_sequences(data, target_column, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i+sequence_length].drop(target_column, axis=1).values)
        y.append(data.iloc[i+sequence_length][target_column])
    return np.array(X), np.array(y)

# Define the target column
target_column = 'Soil Moisture (%)'

# Create sequences for training and testing
sequence_length = 10
X_train, y_train = create_sequences(train_data, target_column, sequence_length)
X_test, y_test = create_sequences(test_data, target_column, sequence_length)

# Build the LSTM model
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Actual Soil Moisture")
plt.plot(y_pred, label="Predicted Soil Moisture")
plt.legend()
plt.title("LSTM Predictions vs Actual")
plt.xlabel("Time Steps")
plt.ylabel("Soil Moisture (%)")
plt.show()
