# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set which contains data from January 1996 to October 2019
dataset_train = pd.read_csv('Train.csv')
training_set = dataset_train.iloc[:, 4:5].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 5885):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#try adding different indicators


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#regressor.compile(optimizer= 'rmsprop', loss= 'mean_squared_error')


# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)#change epochs and batch size

# Part 3 - Making the predictions and visualising the results

# Getting the stock price for the two months November 2019 and December 2019
dataset_test = pd.read_csv('Test.csv')
real_stock_price = dataset_test.iloc[:, 4:5].values

# Getting the predicted stock price for the two months
dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 101):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price= sc.inverse_transform(predicted_stock_price)

pred_array= []
for i in range(len(predicted_stock_price)):
    if(i== 0):
        pred_array.append(1)
    else:
        if(predicted_stock_price[i]> predicted_stock_price[i-1]):
            pred_array.append(1)
        else:
            pred_array.append(0)
            
actual_array= []
for i in range(len(real_stock_price)):
    if(i== 0):
        actual_array.append(1)
    else:
        if(real_stock_price[i]> real_stock_price[i-1]):
            actual_array.append(1)
        else:
            actual_array.append(0)


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real HDFC Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted HDFC Stock Price')
plt.title('HDFC Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('HDFC Stock Price')
plt.legend()
plt.show()