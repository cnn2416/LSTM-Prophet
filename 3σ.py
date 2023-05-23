

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
from pylab import *
import scipy.interpolate as spi
# 3σ原则
def method_1(data):
    data = data['displacement']
    u = data.mean()
    std = data.std()
    stats.kstest(data, 'norm', (u, std))
    print('Mean：%.3f，Std：%.3f' % (u, std))

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(3, 1, 1)
    data.plot(kind='kde', grid=True, style='-k', title=name + 'displacement curve')
    plt.axvline(3 * std, color='r', linestyle="--", alpha=0.8)
    plt.axvline(-3 * std, color='r', linestyle="--", alpha=0.8)
    # 绘制数据密度曲线
    print(name)
    error = data[np.abs(data - u) > 3 * std]
    print(error)
    data_c = data[np.abs(data - u) <= 3 * std]
    # new_x = np.arange(-np.pi, np.pi, 0.1)  # 定义差值点
    ori_x = np.array(data_c.index)
    X = np.arange(0,len(ori_x),1)
    new_x = np.arange(0,len(data.index),1)
    print(new_x,data_c.values)
    ipo3 = spi.splrep(X, data_c.values, k=3)  # 样本点导入，生成参数
    # ipo3 = np.array(ipo3)
    # print(ipo3)
    iy3 = spi.splev(new_x, ipo3)
    ax2 = fig.add_subplot(3, 1, 2)
    plt.scatter(data_c.index, data_c, color='k', marker='.', alpha=0.3)
    plt.scatter(error.index, error, color='r', marker='.', alpha=0.5)

    ax3 = fig.add_subplot(3, 1, 3)
    plt.scatter(data.index, iy3, color='b', marker='.', alpha=0.3)

    plt.grid()
    plt.show()
    # 图表表达
# 箱型图看数据分布情况
# 以内限为界
def method_2(data,data_water):
    data=data['displacement']
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
    data.plot.box(vert=False, grid=True, color=color, ax=ax1)

    s = data.describe()
    print(s)
    print('------')
    # 基本统计量

    q1 = s['25%']
    q3 = s['75%']
    iqr = q3 - q1
    mi = q1 - 1.5 * iqr
    ma = q3 + 1.5 * iqr

    ax2 = fig.add_subplot(2, 1, 2)
    error = data[(data < mi) | (data > ma)]
    data_c = data[(data >= mi) & (data <= ma)]

    plt.scatter(data_c.index, data_c, color='k', alpha=0.3)
    # plt.scatter(error.index, error, color='r', marker='.', alpha=0.5)
    plt.scatter(data_water.index, data_water, color='b',  alpha=0.5)

    # plt.xlim([-10, 10010])
    # plt.grid()


dir_path='../JCK08\\'

dir=os.listdir(dir_path)
num=len(dir)
print(num)
mpl.rcParams["font.sans-serif"] = ["SimHei"]

for name in dir[0:num-1]:
    file_name=dir_path+name
    print(file_name)
    if name =='date40.xlsx' :
        continue
    data=pd.read_excel(file_name,index_col=0)
    # → p(|x - μ| > 3σ) ≤ 0.003
    method_1(data)
    # 箱型图分析
    # method_2(data,data_water)
plt.show()