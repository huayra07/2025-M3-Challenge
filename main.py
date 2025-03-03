import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
file_path = "24 hour heat wave.csv"
df_heat_wave = pd.read_csv(file_path)

# Remove invalid rows (drop rows where 'Time' is not in the correct format)
df_heat_wave = df_heat_wave[df_heat_wave['Time'].str.contains(r'AM|PM', na=False)]

# Convert 'Time' into a proper datetime format
df_heat_wave['Time'] = pd.to_datetime(df_heat_wave['Time'], format='%I:%M %p')

# Add a reference date (assuming data represents a single day)
df_heat_wave['Time'] = df_heat_wave['Time'].apply(lambda x: x.replace(year=2025, month=6, day=1))

# Ensure 'Temperature (°F)' is numeric
df_heat_wave['Temperature (°F)'] = pd.to_numeric(df_heat_wave['Temperature (°F)'], errors='coerce')

# Set 'Time' as the index
df_heat_wave.set_index('Time', inplace=True)

# 🔹 Fix frequency warning by explicitly setting hourly frequency
df_heat_wave = df_heat_wave.asfreq('H')

# Train-test split (first 20 hours for training, last 4 hours for testing)
train_data = df_heat_wave['Temperature (°F)'].iloc[:20]
test_data = df_heat_wave['Temperature (°F)'].iloc[20:]

# Fit ARIMA model (choosing order manually: p=2, d=1, q=2)
model = ARIMA(train_data, order=(2, 1, 2))
results = model.fit()

# Forecast for the test period
forecast_steps = len(test_data)
forecast_values = results.forecast(steps=forecast_steps)

# Plot actual vs. predicted values
plt.figure(figsize=(10, 5))
plt.plot(train_data.index, train_data, label="Train", color='blue')
plt.plot(test_data.index, test_data, label="Test (Actual)", color='orange')
plt.plot(test_data.index, forecast_values, label="Forecast", linestyle="--", color='green')

plt.title("ARIMA Forecast of Temperature")
plt.xlabel("Time")
plt.ylabel("Temperature (°F)")
plt.legend()
plt.grid(True)
plt.show()

# Display forecasted values
print("Forecasted Temperatures:\n", forecast_values)
