import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

window_size = 10  # 窗口大小

df = pd.read_excel('./data/ali.xlsx')
s = df['Open2'].values

result=np.load("./data/RESULT.npy")
pre=np.zeros(shape=(1000))

def RMSE(pred, true):
    return np.mean(np.sqrt(np.square(pred - true)))

def plot(pred, true,i,rmse,MAE):
    plt.title("jubu"+str(i+1))
    plt.plot(true[:200], "r",label="真实值")
    plt.plot(pred[:200], "b",label="预测值")
    plt.text(10,10,"RMSE"+str(rmse))
    plt.text(10,8,"MAE"+str(MAE))
    plt.legend(["zhenshi","yuce"])
    plt.show()
    plt.title("quanju"+str(i+1))
    plt.plot(true, "r",label="真实值")
    plt.plot(pred, "b",label="预测值")
    plt.text(10,10,"RMSE"+str(rmse))
    plt.text(10,8,"MAE"+str(MAE))
    plt.legend(["zhenshi","yuce"])
    plt.show()

scaler = MinMaxScaler()
open_arr = scaler.fit_transform(s.reshape(-1, 1)).reshape(
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

test_label=scaler.inverse_transform(test_label.reshape(-1, 1)).reshape(-1)

pre=np.zeros(shape=test_label.shape)
for i,re in enumerate(result):
    pre+=re
rmse = RMSE(pre, test_label)
MAE = mean_absolute_error(pre, test_label)
print('RMSE ', rmse)
print('RAE ', MAE)
plot(pre, test_label, 12, rmse, MAE)