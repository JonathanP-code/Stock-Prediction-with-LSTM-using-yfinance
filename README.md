# Stock-Prediction-with-LSTM-using-yfinance
This repository contains machine learning models that predict future prices for stocks using LSTM neural networks. The models fetch historical price data from Yahoo Finance, preprocess the data, and generate predictions with buy/sell recommendations. 
Please feel free to recommend or add improvements to this project. 

## Overview
This project uses a Long Short-Term Memory (LSTM) neural network to predict prices based on historical market data retrieved from Yahoo Finance via the `yfinance` library.

<a href="https://numpy.org/"><img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243.svg?logo=numpy&logoColor=white"></a>
<a href="https://www.tensorflow.org/"><img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?logo=tensorflow&logoColor=white"></a>
<a href="https://pypi.org/project/yfinance/"><img alt="yfinance" src="https://img.shields.io/badge/yfinance-003366.svg?logo=python&logoColor=white"></a>
<a href="https://matplotlib.org/"><img alt="matplotlib" src="https://img.shields.io/badge/matplotlib-11557C.svg?logo=python&logoColor=white"></a>
<a href="https://pandas.pydata.org/"><img alt="pandas" src="https://img.shields.io/badge/pandas-150458.svg?logo=pandas&logoColor=white"></a>
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3776AB.svg?logo=python&logoColor=white"></a>

### Features
- Fetches real-time price data using Yahoo Finance (`yfinance`).
- Preprocesses and normalizes data for training.
- Uses an LSTM neural network for time series forecasting.
- Generates predictions for the next 10 days.
- Provides a simple buy/sell/hold recommendation based on predicted vs. actual price.
- Visualizes past price trends and future predictions.

### Installation/ Usage
```bash
pip install numpy pandas matplotlib tensorflow scikit-learn yfinance
```
```python
python gold_price_prediction.py
```

### Data Source and Attribution
This project retrieves data from **Yahoo Finance** via the `yfinance` Python package.
- **Source:** Yahoo Finance (https://finance.yahoo.com/)
- **Attribution Policy:** Data is provided by Yahoo Finance and must be credited accordingly.
- **Disclaimer:** Yahoo Finance provides the data "as is" with no guarantees on accuracy or availability.
- **Usage Limits**: Be mindful of Yahoo Finance's rate limits to avoid access issues.

---

> [!CAUTION]
> This project is **for educational and research purposes only**.
> The predictions and recommendations made by this model **do not constitute financial advice**.
> The author and contributors **are not responsible for any financial decisions** made based on this software.
> Please also view the **Please Note!** file.

> [!IMPORTANT]
> To make the code below functional you will need to replace "STOCK" and "YOURSTOCK" with your actual stock name.
> You may want to take a look at the finished example files given above.

### Example Code

```python code
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

# Function to fetch real-time data
def get_live_STOCK_data(ticker="YOURSTOCK", period="5y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval)
    return df[['Close']]

# Fetch latest price data
symbol = "YOURSTOCK"
df = get_live_STOCK_data(symbol)

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
```

### Example Output
Last Gold Price: 3007.30𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑒𝑑𝑁𝑒𝑥𝑡𝐷𝑎𝑦𝑃𝑟𝑖𝑐𝑒: 3024.12 Recommendation: BUY 📈
![6CC95D77-3C14-4691-8FDE-F34C3E0474C7_1_201_a](https://github.com/user-attachments/assets/2a8d2361-5a7f-4393-b5e0-7bb4fe9e2af0)


### Example Output
Price:  212.79𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑒𝑑𝑁𝑒𝑥𝑡𝐷𝑎𝑦𝑃𝑟𝑖𝑐𝑒: 227.16 Recommendation: BUY 📈
![5BA3D198-C0B5-4708-A876-28209D5B5BBA_1_201_a](https://github.com/user-attachments/assets/80c217c7-d937-49d1-b632-ac02c838b068)


