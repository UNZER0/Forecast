import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

window_size = 10  # 窗口大小
batch_size = 32  # 训练批次大小
epochs = 500  # 训练epoch
filter_nums = 10  # filter数量
kernel_size = 4  # kernel大小


def get_dataset():
    df = pd.read_excel('./ali.xlsx')
    scaler = MinMaxScaler()
    open_arr = scaler.fit_transform(df['Open2'].values.reshape(-1, 1)).reshape(
        -1)  # 归一化，open是首行标题,df['Open'].values.reshape(-1, 1)变成二维数组，每个里边只有一个元素
    X = np.zeros(shape=(
    len(open_arr) - window_size, window_size))  # 0填充的二维数组,总共len(open_arr) - window_size个一维数组，每个数组里边window_size个元素
    label = np.zeros(shape=(len(open_arr) - window_size))  # 0填充的一维数组,里边有len(open_arr) - window_size)个元素
    for i in range(len(open_arr) - window_size):
        X[i, :] = open_arr[i:i + window_size]
        label[i] = open_arr[i + window_size]
    # 前2000个元素为训练集，后1000为测试集
    train_X = X[:2000, :]
    train_label = label[:2000]
    test_X = X[2000:3000, :]
    test_label = label[2000:3000]
    return train_X, train_label, test_X, test_label, scaler
