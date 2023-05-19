import yfinance as yf
from sklearn.linear_model import LogisticRegression
import numpy as np

# Define the ticker symbol for the Indian stock
ticker_symbol = "RELIANCE.NS"

# Define the date range for historical data
start_date = "2022-01-01"
end_date = "2022-12-31"

# Download the historical stock data using yfinance
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Preprocess the data for logistic regression
X = np.array(stock_data["Close"]).reshape(-1, 1)
y = np.where(stock_data["Close"].shift(-1) > stock_data["Close"], 1, 0)
X = X[:-1]  # Remove the last row
y = y[:-1]  # Remove the last row

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Get the user input for the date to make a prediction for
user_input = input("Enter a date to make a prediction for (format: YYYY-MM-DD): ")

# Get the stock data for the user input date
input_data = yf.download(ticker_symbol, start=start_date, end=end_date)
input_close = np.array(input_data["Close"]).reshape(1, -1)

# Convert the input close price to Indian rupees
conversion_rate = 1.0 / stock_data["Close"][0]  # Conversion rate to rupees based on the first closing price
input_close *= conversion_rate

# Use the logistic regression model to make a prediction
prediction = model.predict(input_close)[0]

# Print the predicted result
if prediction == 1:
    print("The predicted stock price for", user_input, "is expected to increase.")
else:
    print("The predicted stock price for", user_input, "is expected to decrease or stay the same.")
