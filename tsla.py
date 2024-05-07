# Import necessary libraries
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
import helper
import pandas as pd
import numpy as np

# Load stock data and Setup datasets
df = pd.read_csv('TSLA.csv')
df = df['Open'].values
df = df.reshape(-1, 1)
dataset_train = np.array(df[:int(df.shape[0] * 0.8)])
dataset_test = np.array(df[int(df.shape[0] * 0.8):])

# Scale the values
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.transform(dataset_test)

# Create train/test datasets
x_train, y_train = helper.create_dataset(dataset_train)
x_test, y_test = helper.create_dataset(dataset_test)

# Reshape the 'x_train' and 'x_test' datasets
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Implement the 'Sequential' model
model = Sequential()
model.add(LSTM(units=4, input_shape=(x_train.shape[1], 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=4, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(x_train, y_train, epochs=5, batch_size=16, verbose=0)

# Predict the values for 'x_test'
predictions = model.predict(x_test)

# Print the array's last column and the model summary
print(predictions[:, -1])
model.summary()
