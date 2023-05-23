#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import xlrd
# import pywt
# import datetime
# import pywt.data
from datetime import datetime
import re
import pandas as pd
from pylab import *
# from scipy import stats
import os
import scipy.interpolate as spi

from fbprophet.plot import add_changepoints_to_plot
from fbprophet import Prophet
filename='..\period\date14.xlsx'
# dir=os.listdir(dir_path)
# num=len(dir)
# print(num)
mpl.rcParams["font.sans-serif"] = ["SimHei"]

matplotlib.rcParams['axes.unicode_minus']=False

data=pd.read_excel(filename,engine='openpyxl')
n=len(data)
for index, t in enumerate(data['Unnamed: 0']):
    dt = datetime.datetime.strptime(t, '%Y-%m-%d')
    data['Unnamed: 0'][index] = dt

# print(data)
df = pd.DataFrame(columns=['ds', 'y'])
# 调整时间为date形式
df['ds']=data[:int(n * 0.90)]['Unnamed: 0']


df['y']=data[:int(n * 0.90)]['trend']
# m = Prophet(changepoint_prior_scale=0.5,n_changepoints=30,changepoint_range=0.8, growth = "logistic")
m = Prophet()
m.fit(df)
# test
df_test = pd.DataFrame(columns=['ds', 'y'])
df_test['ds'] = df['ds']
df_test['y'] = data['trend']

future = m.make_future_dataframe(periods = int(n * 0.1), freq = 'd')

forecast = m.predict(future)
forecast[['ds', 'yhat']].tail()

# 保存
# forecast.to_excel(name)
# plt.plot(df_test['ds'],data['trend'],color='black')
plt.plot(data['Unnamed: 0'], data['trend'], color='green')
plt.plot(forecast['ds'][int(n * 0.90) :],forecast["yhat"][int(n * 0.90) :],color='red')
df = pd.DataFrame(columns=['ds', 'yearly'])


df['ds'] = forecast['ds'][int(n * 0.90) :]
df['yearly'] = forecast[["yhat"]][int(n * 0.90) :]
name = "20230224-JCK08-14-trend.xlsx"
df.to_excel(name)

# plt.plot(forecast['ds'], forecast['additive_terms'], color='red')
# plt.title('周期、趋势、预测'+name)
plt.xlabel('time')
plt.ylabel('displacement/mm')
plt.legend(['prophet','real'])
# plt.axvline(forecast['ds'][int(n * 0.95) + 2], color='red', linestyle='--')
plt.grid(axis='both')  # 此参数为默认参数
plt.show()















































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































