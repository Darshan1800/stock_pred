import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Define the ticker symbol for the stock
ticker_symbol = "HDFC.NS"

# Define the date range for historical data
start_date = "2000-01-01"
end_date = "2023-05-18"

# Define the target date for prediction
target_date = "2023-05-19"

# Download the historical stock data using yfinance
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Extract the closing prices from the fetched data
closing_prices = np.array(stock_data["Close"]).reshape(-1, 1)

# Scale the closing prices using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(closing_prices)

# Prepare the training data
X_train = []
y_train = []

# Define the sequence length for each training example
sequence_length = 10

for i in range(sequence_length, len(scaled_prices)):
    X_train.append(scaled_prices[i - sequence_length:i, 0])
    y_train.append(scaled_prices[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the input data for LSTM (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
model.fit(X_train, y_train, epochs=10, batch_size=8)

# Get the stock data for the target date
target_data = yf.download(ticker_symbol, start=target_date)
target_price = np.array(target_data["Close"]).reshape(-1, 1)

# Scale the target price using MinMaxScaler
scaled_target_price = scaler.transform(target_price)

# Prepare the input data for prediction
X_predict = scaled_prices[-sequence_length:]
X_predict = np.reshape(X_predict, (1, X_predict.shape[0], 1))

# Make a prediction for the target date
predicted_price = model.predict(X_predict)

# Inverse scale the predicted price
predicted_price = scaler.inverse_transform(predicted_price)

# Print the predicted closing price
print("Predicted closing price for", target_date, ":", predicted_price[0][0])
