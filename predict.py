import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tcn.tcn import TCN
from tensorflow import keras

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from train import IMF
from train import s


window_size = 10  # 窗口大小
batch_size = 32  # 训练批次大小
epochs = 5  # 训练epoch
filter_nums = 10  # filter数量
kernel_size = 4  # kernel大小
RESULT = np.zeros(shape=(IMF.shape[0], 1000))

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

def load_model():

    for i, imf in enumerate(IMF):
        train_X, train_label, test_X, test_label, scaler = get_dataset(IMF[i])
        model=keras.models.load_model("./data/model"+str(i+1)+".h5",custom_objects={'TCN': TCN})
        model.load_weights("./data/model_weights"+str(i+1)+".h5")
        evaluate=model.evaluate(test_X, test_label)
        prediction = model.predict(test_X)
        scaled_prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(-1)
        # if(i<3):
        #     RESULT[i]=scaled_prediction;
        scaled_test_label = scaler.inverse_transform(test_label.reshape(-1, 1)).reshape(-1)
        print("第"+str(i+1)+"个分量误差")
        print(evaluate)
        rmse=RMSE(scaled_prediction, scaled_test_label)
        MAE=mean_absolute_error(scaled_prediction, scaled_test_label)
        print('RMSE ', rmse)
        print('RAE ',MAE )
        plot(scaled_prediction, scaled_test_label,i,rmse,MAE)
    np.save("./data/RESULT",RESULT)

if __name__ == '__main__':
    load_model()