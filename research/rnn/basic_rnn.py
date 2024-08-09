import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class BasicRNN:
    def __init__(self):
        self.rnn = keras.models.Sequential()
        self.dataset_train = pd.read_csv('dataset/Google_Stock_Price_Train.csv')
        self.dataset_test = pd.read_csv('dataset/Google_Stock_Price_Test.csv')
        self.min_max_scaler = MinMaxScaler()

        self.training_set = self.dataset_train.iloc[:, 1:2].values
        self.min_max_scaler.fit(self.training_set)

    def create_rnn(self):
        training_set_scaled = self.min_max_scaler.transform(self.training_set)

        x_train = []
        y_train = []

        for i in range(60, len(training_set_scaled)):
            x_train.append(training_set_scaled[i - 60: i, 0])
            y_train.append(training_set_scaled[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        self.rnn = keras.models.Sequential()
        self.rnn.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self.rnn.add(keras.layers.Dropout(0.2))
        self.rnn.add(keras.layers.LSTM(units=50, return_sequences=True))
        self.rnn.add(keras.layers.Dropout(0.2))
        self.rnn.add(keras.layers.LSTM(units=50, return_sequences=True))
        self.rnn.add(keras.layers.Dropout(0.2))
        self.rnn.add(keras.layers.LSTM(units=50))
        self.rnn.add(keras.layers.Dropout(0.2))
        self.rnn.add(keras.layers.Dense(units=1))

        self.rnn.compile(optimizer='adam', loss='mean_squared_error')
        self.rnn.fit(x_train, y_train, epochs=100, batch_size=32)

        keras.models.save_model(self.rnn, 'rnn.keras')

    def load_rnn(self):
        self.rnn = keras.models.load_model('rnn.keras')

    def predict(self, x):
        x = self.min_max_scaler.transform(x)
        result = self.rnn.predict(x)
        return result

    def run_test(self):
        real_stock_price = self.dataset_test.iloc[:, 1:2].values
        dataset_total = pd.concat((self.dataset_train['Open'], self.dataset_test['Open']), axis=0)
        inputs = dataset_total[len(dataset_total) - len(self.dataset_test) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = self.min_max_scaler.transform(inputs)
        X_test = []
        for i in range(60, 80):
            X_test.append(inputs[i - 60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = self.rnn.predict(X_test)
        predicted_stock_price = self.min_max_scaler.inverse_transform(predicted_stock_price)

        plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
        plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
        plt.title('Google Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Google Stock Price')
        plt.legend()
        plt.show()


basic_rnn = BasicRNN()
# basic_rnn.create_rnn()
basic_rnn.load_rnn()
basic_rnn.run_test()

# test_data = np.array([
#     78.81,
#     88.36,
#     86.08,
#     95.26,
#     06.4,
#     807.86,
#     805,
#     807.14,
#     807.48,
#     807.08,
#     805.81,
#     805.12,
#     806.91,
#     807.25,
#     822.3,
#     829.62,
#     837.81,
#     834.71,
#     814.66,
#     796.86,
# ])
# test_data = np.reshape(test_data, (-1, 1))
# print(basic_rnn.predict(test_data))
