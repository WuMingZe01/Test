# LSTM，多特征值预测基于用户情感倾向

import requests
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from urllib.parse import urlencode
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def split_sequences(X: np.ndarray, time_steps: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    '''
    生成输入输出序列

    参数
    ---
    X : m 行 n 列 array ，其中最后一列为待输出的输出，前面的均为待输入的数据
    time_steps : 时间步长，即使用前 time_steps 给数据来预测下一个数据

    返回
    ---
    生成的 X 和 y array

    '''
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    length = len(X)
    n_features = X.shape[1]-1
    _X = []
    _y = []
    for start_idx in range(length):
        end_idx = start_idx+time_steps
        if end_idx >= length:
            break
        _X.append(X[start_idx:end_idx, :-1])
        _y.append(X[end_idx-1, -1])
    _X = np.array(_X)
    _X = _X.reshape((-1, time_steps, n_features))
    _y = np.array(_y)
    return (_X, _y)




# 读取股票信息表格
df = pd.read_csv('fund_Price_fill.csv')
# 删除缺少值
df = df.dropna()

# 指定研究的特征（最后一个为待预测的数据!）
# 全部的 features ：['开盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率','收盘']
features = ['累计净值','情绪指数','日增长率','单位净值']

# 创建一个 MinMaxScaler 对象，方便将数据归一化
# 归一化公式：x =  (x - min)/(max-min)
sc = MinMaxScaler(feature_range=(0, 1))

# 将数据转为 numpy 数组
values = df[features].to_numpy()

# 归一化 values
scaled_values = sc.fit_transform(values)

# 生成序列
X, y = split_sequences(scaled_values)

# 训练数据个数
train_size = int(len(df)*0.6)

# 训练集
X_train = X[:train_size]
y_train = y[:train_size]

# 测试集
X_test = X[train_size:]
y_test = y[train_size:]

# 创建模型
model = Sequential()
# 添加 含有 50 个单元的 LSTM 网络(第一层)
model.add(LSTM(50,return_sequences = True))
# 添加 含有 30 个单元的 LSTM 网络(第二层)
model.add(LSTM(30,return_sequences = True))
# 添加 含有 10 个单元的 LSTM 网络(第三层)
# 注意，最后一层没有 return_sequences = True ！！！
model.add(LSTM(10))


# 添加输出层网络以输出预测的股票收盘价格
model.add(Dense(1))
# 编译模型
model.compile(loss='mae', optimizer='adam')
# 拟合模型
model.fit(X_train, y_train, epochs=30,validation_split=0.2)
# 真实收盘价格
y_real = values[-len(y_test):, -1]
plt.plot(y_real, label='真实基金价格')

y_p = model.predict(X_test).reshape(-1, 1)
# 将全部收盘价归一化
sc.fit_transform(df['单位净值'].to_numpy().reshape(-1, 1))

# 归一化逆过程，即将归一化的数据转为真实数据
# 预测的收盘价格
y_p = sc.inverse_transform(y_p.reshape(-1, 1))
# 绘图
plt.plot(y_p, label='预测基金价格')
# 绘制图例
plt.legend()
# 绘制标题
plt.title('根据 {} 预测 {} 的基金走势'.format("、".join(features[:-1]), 110022))
# 保存图片
plt.savefig(f'110022.jpg')
# 显示图像
plt.show()