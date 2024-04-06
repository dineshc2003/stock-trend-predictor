import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data

import distutils
import keras
#pip install setuptools
#pip install tensorflow
from keras.models import load_model
import streamlit as st
import yfinance as yf
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from pandas_datareader import data as pdr
import scikit-learn

yf.pdr_override()



start = '2010-01-01'
end = '2019-12-31'
st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = pdr.get_data_yahoo(user_input, start, end)

#Describing Data
st.subheader('Data from 2010-2019')
st.write(df.describe())
 
#visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)
#ma=moving average
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df['Close'].rolling(window=100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df['Close'].rolling(window=100).mean()
ma200 = df['Close'].rolling(window=200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)
x_train = []
y_train = []
for i in range(100, data_training_array.shape[0]):
  x_train.append(data_training_array[i-100:i])
  y_train.append(data_training_array[i,0])


x_train, y_train = np.array(x_train), np.array(y_train)


#splitting data into x_train & y_train


#load my model



model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 80, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))
model.compile(optimizer='adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 50)

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days,data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

