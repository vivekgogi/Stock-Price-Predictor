import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

def load_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data.set_index('Date').tz_localize(None)
    return stock_data

def lstm_predict(stock_data):
    data = stock_data.filter(["Close"])
    dataset = data.values
    train_len = int(np.ceil(len(dataset) * 0.8))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[:train_len, :]

    seq_len = 60
    x_train, y_train = [], []

    for i in range(seq_len, len(train_data)):
        x_train.append(train_data[i - seq_len : i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="linear"))

    model.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"]
    )
    model.fit(x_train, y_train, batch_size=32, epochs=5)

    if train_len == 0:
        train_len = len(scaled_data) - len(train_data)

    test_data = scaled_data[train_len - seq_len :, :]
    x_test = []

    for i in range(seq_len, len(test_data)):
        x_test.append(test_data[i - seq_len : i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    predict_prices = [price[0] for price in predictions.tolist()]

    return predict_prices
