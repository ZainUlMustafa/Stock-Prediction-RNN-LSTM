'''AUTHOR: ZAIN UL MUSTAFA'''
'''http://www.github.com/ZainUlMustafa'''

print('STOCK PREDICTION USING RNN LSTM')
import numpy as np
import pandas as pd
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import  Dropout
from keras.models import model_from_json
from keras.models import load_model

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
fmt = '$%.0f'
tick = mtick.FormatStrFormatter(fmt)

import stockproc

#########################################################################
'''Path and filename'''

path = 'Data/KSE/'
# make the boolean false if you want to read data offline and true for online from Quandl
data_csv = stockproc.getStockData(path,'hbl',True)
data_csv[['Last Day Close']].plot()
plt.show()
plt.clf()
#########################################################################
'''Defining how much data to use'''
# Data to be used
# The more frequent this is, the better

percentage_of_data = 1.0
data_to_use = int(percentage_of_data*(len(data_csv)-1))

# 80% of data will be of training
train_end = int(data_to_use*0.8)

total_data = len(data_csv)
print("total_data:", total_data)

#########################################################################
'''Making a dataset'''
# Start from 0
start = total_data - data_to_use

# Currently doing prediction only for 1 step ahead
steps_to_predict = 1

yt,yt1,yt2,yt3,vt = stockproc.feature_engineering(start,total_data,data_csv)
# Order -> 5,2,3,4,6

print("yt head :")
print(yt.head())

#########################################################################
'''Shifting the closed price column by 1'''
yt_ = yt.shift(-1)

data = pd.concat([yt, yt_, vt, yt1, yt2, yt3], axis=1)
data.columns = ['yt', 'yt_', 'vt', 'yt1', 'yt2', 'yt3']
     
data = data.dropna()
     
print(data)

#########################################################################
'''Renaming the columns'''     
# target variable - closed price
# after shifting
y = data['yt_']

#   closed, volume, open, high, low    
cols = ['yt', 'vt', 'yt1', 'yt2', 'yt3']
x = data[cols]

#########################################################################
'''Preprocessing the data'''
scaler_x = preprocessing.MinMaxScaler (feature_range=(-1, 1))
x = np.array(x).reshape((len(x) ,len(cols)))
x = scaler_x.fit_transform(x)

scaler_y = preprocessing.MinMaxScaler (feature_range=(-1, 1))
y = np.array (y).reshape ((len( y), 1))
y = scaler_y.fit_transform (y)

#########################################################################
'''Making the train and test dataset'''
X_train = x[0 : train_end,]
X_test = x[train_end+1 : len(x),]    
y_train = y[0 : train_end] 
y_test = y[train_end+1 : len(y)]  

X_train = X_train.reshape (X_train. shape + (1,)) 
X_test = X_test.reshape(X_test.shape + (1,))

#########################################################################
'''RNN and LSTM model'''
batch_size = 32
nb_epoch = 25
neurons = 512

seed = 2016
np.random.seed(seed)
model = Sequential ()
model.add(LSTM(neurons, return_sequences=True, activation='tanh', inner_activation='hard_sigmoid', input_shape=(len(cols), 1)))
model.add(Dropout(0.2))
model.add(LSTM(neurons, return_sequences=True,  activation='tanh'))
model.add(Dropout(0.2))
model.add(LSTM(neurons, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(output_dim=1, activation='linear'))
model.add(Activation('tanh'))

model.compile(loss='mean_squared_error' , optimizer='adam')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.2)

print(model.summary())

#########################################################################
'''Calculating the score'''
score_train = model.evaluate(X_train, y_train, batch_size =1)
score_test = model.evaluate(X_test, y_test, batch_size =1)
print("in train MSE = ", round( score_train ,4)) 
print("in test MSE = ", score_test )

#########################################################################
'''Saving the model'''
# serialize model to JSON
model_json = model.to_json()
with open("model_adam25e.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_adam25e.h5")
print("Saved model to disk")

#########################################################################
'''Printing the predictions'''
pred = model.predict(X_test) 
pred = scaler_y.inverse_transform(np.array(pred).reshape((len(pred), 1)))

prediction_data = pred[-1]     

model.summary()
print ("Inputs: {}".format(model.input_shape))
print ("Outputs: {}".format(model.output_shape))
print ("Actual input: {}".format(X_test.shape))
print ("Actual output: {}".format(y_test.shape))

print ("prediction data:")
print (prediction_data)

print ("actual data")
X_test = scaler_x.inverse_transform(np.array(X_test).reshape((len(X_test), len(cols))))
print (X_test)

#########################################################################
'''Plotting'''
plt.plot(pred, label="predictions")

y_test = scaler_y.inverse_transform(np.array(y_test).reshape((len( y_test), 1)))
plt.plot([row[0] for row in y_test], label="actual")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)

ax = plt.axes()
ax.yaxis.set_major_formatter(tick)
plt.show()
plt.clf()

#########################################################################
