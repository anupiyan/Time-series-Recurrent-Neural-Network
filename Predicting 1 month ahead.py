from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,SimpleRNN
from math import sqrt
import math
from matplotlib import pyplot
from numpy import split
import numpy
from numpy import array
from numpy import mean
from numpy.linalg import eig
# split a univariate dataset into train/test sets
def split_dataset(data):
	# split into standard weeks
	train, test = data[:450], data[450:]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/30))
	test = array(split(test, len(test)/30))
	return train, test

#reading the file
def reading_data():
    #reading the data
    dataframe = read_csv('block 36 Mean value per day.csv',  usecols=[2], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset = scaler.fit_transform(dataset)
    print(dataset.ndim)
    return dataset,scaler

#creating testing and training datasets
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# train the model
def model_train(train_dataX,train_dataY,look_back=1):
        model = Sequential()
        model.add(SimpleRNN(100, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(train_dataX, train_dataY, epochs=300, batch_size=20, verbose=1)
        return model
#testing the model
def model_testing(model,test_dataX,test_dataY,scaler):
        testPredict = model.predict(test_dataX)
        # invert predictions
        print("The test_dataY "+str(testPredict.ndim))
        print(test_dataY.shape)
        test_dataY = numpy.asarray(test_dataY)
        test_dataY = test_dataY.reshape(3)
        test_dataY = scaler.inverse_transform([test_dataY])
        testPredict = scaler.inverse_transform(testPredict)
        #MAPE
        testScore = numpy.mean(numpy.abs((test_dataY-testPredict)/test_dataY))*100
        print('Test MAPE %.2f'%(testScore))
        
data,scaler = reading_data()
train,test = split_dataset(data)
train_dataX,train_dataY= create_dataset(train)
test_dataX,test_dataY = create_dataset(test)
model = model_train(train_dataX,train_dataY)
model_testing(model,test_dataX,test_dataY,scaler)
