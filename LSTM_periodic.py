#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# import xlrd
import pywt
from datetime import datetime
import pywt.data
import re
import pandas as pd
from pylab import *
from scipy import stats

import os
import datetime
import scipy.interpolate as spi


def datetransform(data):
    list = data.index
    array = []
    for index, i in enumerate(list):
        time = pd.to_datetime(i)
        array.append(time)
        # print(array)
    data.index = array

    data.sort_index()
    print(data)
    data.to_excel(file_name)

def da_sort(data):
    data.sort_index(inplace=True)
    print(data)
    data.to_excel(file_name)

from pandas import DataFrame
from pandas import concat

from sklearn.metrics import mean_squared_error
from datetime import datetime
import tensorflow as tf
from math import sqrt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Model, load_model, model_from_json, Sequential
from tensorflow.keras.layers import Dense,Conv1D,MaxPooling1D,Flatten,RepeatVector,TimeDistributed
from tensorflow.keras.layers import LSTM
physical_devices = tf.config.experimental.list_physical_devices('GPU')

def out_error(data):
    u = data.mean()  #
    std = data.std()  #
    stats.kstest(data, 'norm', (u, std))
    print('Mean：%.3f，Std：%.3f' % (u, std))

    error = data[np.abs(data - u) > 3 * std]
    data_c = data[np.abs(data - u) <= 3 * std]
    return data_c
def cs_to_sl(file_name):
    # load data
    dataset = pd.read_excel(file_name,index_col=0,engine='openpyxl')
    # 选择训练数据
    dataset = dataset.drop('trend', axis=1)
    dataset = dataset.drop('yhat', axis=1)
    # dataset = dataset.drop('降雨', axis=1)
    # dataset = dataset.drop('水位', axis=1)
    dataset = dataset.drop('pre', axis=1)
    # dataset = dataset.drop('ori', axis=1)
    dataset = dataset.drop('sun', axis=1)
    dataset = dataset.drop('yearly', axis=1)

    # for index,t in enumerate(dataset['Unnamed: 0']):
    #     dt = datetime.strptime(t, '%Y-%m-%d')
    #     # print(dt)
    #     dataset['Unnamed: 0'][index]=datetime.timestamp(dt)*1000

    values = dataset.values
    values = values.astype('float32')
    # normalize feature
    scaler = MinMaxScaler(feature_range=(0, 1))
    # print(scaler)
    scaled = scaler.fit_transform(values)
    return scaler, scaled


def train_test(reframed,look_back,out_step):
    # split into train & test sets
    n=len(reframed)
    values = reframed
    # 数据集划分
    # train = values[:int(n*0.8)]
    # test = values[int(n*0.8):]

    train = values[:int(n*0.95)]
    test = values[int(n*0.95):]

    # split into input and outputs
    # look_back represent timesept
    # look_back = 20
    # out_step = 10
    trainX, train_y = create_dataset(train, look_back,out_step)
    testX, test_y = create_dataset(test, look_back,out_step)
    # train_X, train_y = train[:,:17], train[:,17:]
    #样本属于为 [samples, timesteps, features]
    train_X = trainX.reshape((trainX.shape[0], trainX.shape[1],trainX.shape[2]))
    print(train_X.shape)

    test_X = testX.reshape((testX.shape[0], testX.shape[1],trainX.shape[2]))

    return train_X, train_y, test_X, test_y

def fit_network(train_X, train_y, test_x, test_y, scaler,step_out):
    # print(train_X.shape,train_X)
    model = Sequential()
    n_features=train_X.shape[2]
    n_steps_in=train_X.shape[1]
    n_steps_out=step_out
    model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))

    # model.add(TimeDistributed(Dense(n_features)))
    # lstm_cnn.compile(loss='mae', optimizer='adam')
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    # fit network
    history = model.fit(train_X, train_y, epochs=100, batch_size=10, validation_data=(test_x, test_y), verbose=1,
                        shuffle=False)
    model.save('LSTMv1-rain-undergroud.h5')
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
def create_dataset(dataset, look_back,out_step):

    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-out_step):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back:i + look_back+out_step])
    return np.array(dataX),np.array(dataY)
def evaluate(train_X, train_y, test_x, test_y, scaler,values,list):
    n = len(list)
    model=load_model('LSTMv1-rain-undergroud.h5')
    yhat = model.predict(test_x)
    yhat = yhat.reshape((-1, yhat.shape[2]))
    test_y = test_y.reshape((-1, test_y.shape[2]))
    # 反归一化
    # for i in range(len(yhat)):
    yhat = scaler.inverse_transform(yhat)[:,0]
    test_y = scaler.inverse_transform(test_y)[:,0]
    plt.axvline(data.index[int(n * 0.95)+2], color='red', linestyle='--')
    print(yhat)
    m=len(test_y)
    plt.plot(data[-m:].index,yhat,color='red',label='yhat')
    plt.plot(data[int(n*0.95)+2:].index,test_y,color='blue',label='true')
    plt.plot(list.index,list['ori'], color='black', label='ori')

    plt.title('各深度位移变化曲线预测')
    plt.legend(loc='upper left')
    plt.show()
    # calculate RMSE
    rmse = sqrt(mean_squared_error(yhat, test_y))
    print('test RMSE: {}'.format(rmse))


    # 生成新文件
    df = pd.DataFrame(columns=['ds', 'pre','ori'])
    df['ds'] = list.index
    df['pre'] = list['ori']
    df['pre'][int(n * 0.95)+2:] = yhat
    df['ori'] = list['ori']
    name='date14-lstm.xlsx'
    # df.to_excel(name)


# print(num)
mpl.rcParams["font.sans-serif"] = ["SimHei"]


# file_name='E:\个人\研究内容\data_everyday-4\JCK08\\date14.xlsx'
file_name='..\period\date14.xlsx'

data=pd.read_excel(file_name,index_col=0,engine='openpyxl')
array = []
for index, t in enumerate(data.index):
    dt = datetime.strptime(t, '%Y-%m-%d')
    array.append(dt)
data.index=array

scaler, values = cs_to_sl(file_name)
# n_hours = 4
look_back=1
out_step=1
# train_test(reframed,n_hours)
train_X, train_y, test_X, test_y = train_test(values,look_back,out_step)
fit_network(train_X, train_y, test_X, test_y, scaler,out_step)
evaluate(train_X, train_y, test_X, test_y, scaler, values,data)








































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































