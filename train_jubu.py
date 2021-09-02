from __future__ import division, print_function
import pylab as plt
from PyEMD import EMD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tcn.tcn import TCN
from tensorflow import keras
from train import IMF


'超参数定义'
window_size = 10  # 窗口大小
batch_size = 32  # 训练批次大小
epochs = 100  # 训练epoch
filter_nums = 4  # filter数量
kernel_size = 2  # kernel大小



def get_dataset(imf):
    scaler = MinMaxScaler()
    open_arr = scaler.fit_transform(imf.reshape(-1, 1)).reshape(-1)   #归一化，open是首行标题,df['Open'].values.reshape(-1, 1)变成二维数组，每个里边只有一个元素
    X = np.zeros(shape=(len(open_arr) - window_size, window_size))      #0填充的二维数组,总共len(open_arr) - window_size个一维数组，每个数组里边window_size个元素
    label = np.zeros(shape=(len(open_arr) - window_size))               #0填充的一维数组,里边有len(open_arr) - window_size)个元素
    for i in range(len(open_arr) - window_size):
        X[i, :] = open_arr[i:i + window_size]
        label[i] = open_arr[i + window_size]
    #前2000个元素为训练集，后1000为测试集
    train_X = X[:2000, :]
    train_label = label[:2000]
    test_X = X[2000:3000, :]
    test_label = label[2000:3000]
    return train_X, train_label, test_X, test_label, scaler

'模型训练'
def build_model():
    train_X, train_label, test_X, test_label, scaler = get_dataset(IMF[0])
    model = keras.models.Sequential([
        keras.layers.Input(shape=(window_size, 1)),
        TCN(nb_filters=filter_nums,  # 滤波器的个数，类比于units
            kernel_size=kernel_size,  # 卷积核的大小
            dilations=[1,2,4,8]),  # 空洞因子
        keras.layers.Dense(units=1, activation='relu')
    ])
    model.summary()         #模型展示
    model.compile(optimizer='adam', loss='mae', metrics=['mae']) #编译
    model.fit(train_X, train_label, validation_split=0.2, epochs=epochs)   #拟合

    # 保存参数
    model.save("./data/model"+str(0+1)+".h5")
    model.save_weights("./data/model_weights"+str(0+1)+".h5")

if __name__ == '__main__':
    build_model()