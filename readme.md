# TSLA Stock Prediction using Neural Networks

This repository contains code for predicting TSLA (Tesla, Inc.) stock prices using LSTM (Long Short-Term Memory) neural networks. The model is built using Python and Keras.

This will train the LSTM model on the TSLA stock data and make predictions for the test set.

## Overview

The code in `tsla_prediction.py` follows these main steps:

1. **Importing Necessary Libraries**: Importing required libraries such as NumPy, Pandas, and Keras.

2. **Loading and Preprocessing Data**: Loading TSLA stock data from a CSV file, scaling the data using MinMaxScaler, and splitting it into training and testing datasets.

3. **Creating Train/Test Datasets**: Creating input and output sequences for the LSTM model.

4. **Building the Model**: Constructing a Sequential model with LSTM layers and a Dense output layer.

5. **Compiling the Model**: Configuring the model for training with an optimizer and loss function.

6. **Training the Model**: Fitting the model to the training data.

7. **Making Predictions**: Using the trained model to predict TSLA stock prices for the test set.

8. **Displaying Results**: Printing the predicted stock prices and the model summary.

## Model Architecture

The neural network model consists of:

- Two LSTM layers with 4 units each and a dropout rate of 0.2.
- A Dense output layer with 1 unit.

The model is compiled using the Adam optimizer and Mean Squared Error loss function.

