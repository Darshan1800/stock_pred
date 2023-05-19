import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Define the ticker symbol for the Indian stock
ticker_symbol = "ZOMATO.NS"

# Define the date range for historical data
start_date = "2000-01-01"
end_date = "2023-05-18"

# Download the historical stock data using yfinance
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)

print("Stock data\n",stock_data)
# Preprocess the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(stock_data["Close"]).reshape(-1, 1))

# Split the data into training and testing sets
train_data = scaled_data[:int(0.8 * len(scaled_data))]
test_data = scaled_data[int(0.8 * len(scaled_data)):]

# Create the training data sequences and labels
train_sequences = []
train_labels = []
sequence_length = 10  # Define the sequence length for each training example

for i in range(sequence_length, len(train_data)):
    train_sequences.append(train_data[i - sequence_length:i, 0])
    train_labels.append(1 if train_data[i, 0] > train_data[i - 1, 0] else 0)

train_sequences = np.array(train_sequences)
train_labels = np.array(train_labels)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_sequences.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_sequences, train_labels, epochs=10, batch_size=32)

# Create the testing data sequences and labels
test_sequences = []
test_labels = []

for i in range(sequence_length, len(test_data)):
    test_sequences.append(test_data[i - sequence_length:i, 0])
    test_labels.append(1 if test_data[i, 0] > test_data[i - 1, 0] else 0)

test_sequences = np.array(test_sequences)
test_labels = np.array(test_labels)

# Make predictions on the testing data
predictions = model.predict(test_sequences)
predictions = np.round(predictions).flatten()

# Generate the classification report
report = classification_report(test_labels, predictions)
print(report)

# Get the user input for the date to make a prediction for
# user_input = input("Enter a date to make a prediction for (format: YYYY-MM-DD): ")

# Get the stock data for the user input date
start_date = "2023-05-20"
input_data = yf.download(ticker_symbol,progress=False,start=start_date )
input_close = scaler.transform(np.array(input_data["Close"]).reshape(-1, 1))
input_sequence = np.array([input_close[-sequence_length:, 0]])



# Make a prediction for the user input date
input_prediction = model.predict(input_sequence)

predicted_price = scaler.inverse_transform(input_prediction)
print("Prediction data =",predicted_price)
input_prediction = np.round(input_prediction).flatten()[0]

# Print the predicted result
if input_prediction == 1:
    print("The predicted stock price for is expected to increase.")
else:
    print("The predicted stock price for is expected to decrease or stay the same.")

