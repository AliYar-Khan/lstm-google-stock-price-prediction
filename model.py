from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
training_set = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = training_set.iloc[:, 1:2].values


scaler = MinMaxScaler()

training_set = scaler.fit_transform(training_set)


# getting inputs and outputs
x_train = training_set[0:-1]
y_train = training_set[1: training_set.shape[0]]

# Reshape
x_train = np.reshape(x_train, (x_train.shape[0], 1, 1))

# Building the RNN
regressor = Sequential()
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))
regressor.add(Dense(units=1))

# compile the RNN
regressor.compile(optimizer='rmsprop', loss='mean_squared_error')

# fit model to training set
regressor.fit(x_train, y_train, epochs=400, batch_size=32)

# import dataset
test_set = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = test_set.iloc[:, 1:2].values

# making the predictions
inputs = real_stock_price


inputs = scaler.transform(inputs)

# [samples, time_steps, features]
# Samples - This is the len(inputs), or the amount of data points you have.

# Time steps - This is equivalent to the amount of time steps you run your recurrent
# neural network. If you want your network to have memory of 60 characters,
# this number should be 60.

# Features - this is the amount of features in every time step.
# If you are processing pictures, this is the amount of pixels.
# In this case you seem to have 1 feature per time step.

inputs = np.reshape(inputs, (inputs.shape[0], 1, 1))
predicted_stock_price = regressor.predict(inputs)


# Inverse the scaler
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


plt.plot(real_stock_price, color='red', label='Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='LSTM Stock Price')
plt.title("GOOGLE STOCK PRICE PREDICTION")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
