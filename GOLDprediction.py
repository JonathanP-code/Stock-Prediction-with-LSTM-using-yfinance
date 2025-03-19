import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import datetime

# Function to fetch real-time GOLD stock data
def get_live_gold_data(ticker="GC=F", period="5y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    return df[['Close']]

# Fetch latest Gold price data
symbol = "GC=F"
df = get_live_gold_data(symbol)

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Prepare training data
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60  # Use last 60 days for prediction
X, y = create_sequences(scaled_data, seq_length)

# Train-test split
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation="relu"),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

# Function to predict future prices
def predict_future(days=10):
    future_predictions = []
    last_sequence = X_test[-1]  # Start with the most recent sequence

    for _ in range(days):
        next_prediction = model.predict(last_sequence.reshape(1, seq_length, 1))[0][0]  # Predict next price
        future_predictions.append(next_prediction)

        # Shift window forward
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[-1] = next_prediction  # Add predicted value at the end

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()  # Convert back to actual prices

# Predict next 10 days
future_prices = predict_future(10)

# Generate Buy/Sell recommendation
last_price = df.iloc[-1, 0]
predicted_next_day = future_prices[0]
recommendation = "BUY 📈" if predicted_next_day > last_price else "SELL 📉" if predicted_next_day < last_price else "HOLD ⚖️"

print(f"Last Gold Price: ${last_price:.2f}")
print(f"Predicted Next Day Price: ${predicted_next_day:.2f}")
print(f"Recommendation: {recommendation}")

# Plot last 30 days + 10-day prediction
plt.figure(figsize=(12, 6))
plt.plot(df.index[-30:], df['Close'].iloc[-30:], label="Last 30 Days Prices", color="blue")
plt.axvline(df.index[-1], color="red", linestyle="dashed", label="Today")
future_dates = pd.date_range(df.index[-1], periods=10, freq="D")
plt.plot(future_dates, future_prices, label="Next 10 Days Prediction", color="orange")
plt.legend()
plt.title("Gold Price Prediction (Last 30 Days + Next 10 Days)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.show()
