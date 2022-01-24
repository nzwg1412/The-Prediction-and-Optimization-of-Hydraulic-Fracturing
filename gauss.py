import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import copy


def gauss_noisy(x, y):
    """
    对输入数据加入高斯噪声
    :param x: x轴数据
    :param y: y轴数据
    :return:
    """
    mu = 0
    sigma = 0.05
    for i in range(len(x)):
        x[i] += random.gauss(mu, sigma)
        y[i] += random.gauss(mu, sigma)


if __name__ == '__main__':
    data = pd.read_csv('6_stages.csv')
    data.dropna(axis=0, how='any', inplace=True)
    data1 =  copy.deepcopy(data)
    data2 =  copy.deepcopy(data)
    noise = 0.1
    # if noise:
    #     data1[['Treatment Pressure of Stage 1', 'Treatment Pressure of Stage 2', 'Treatment Pressure of Stage 3', 'Treatment Pressure of Stage 4', 'Treatment Pressure of Stage 5', 'Treatment Pressure of Stage 6', 'Treatment Volume of Stage 1', 'Treatment Volume of Stage 2', 'Treatment Volume of Stage 3', 'Treatment Volume of Stage 4', 'Treatment Volume of Stage 5', 'Treatment Volume of Stage 6','Fracture Spacing']] = data1[['Treatment Pressure of Stage 1', 'Treatment Pressure of Stage 2', 'Treatment Pressure of Stage 3', 'Treatment Pressure of Stage 4', 'Treatment Pressure of Stage 5', 'Treatment Pressure of Stage 6', 'Treatment Volume of Stage 1', 'Treatment Volume of Stage 2', 'Treatment Volume of Stage 3', 'Treatment Volume of Stage 4', 'Treatment Volume of Stage 5', 'Treatment Volume of Stage 6','Fracture Spacing']].apply(lambda x: (x + noise * random.gauss(0, np.std(x, axis=0))))
    #     noise_data = np.array(data2['Fracture Spacing'])
    #     noise_data = noise_data + noise * \
    #                  np.std(noise_data) * np.random.randn(noise_data.shape[0])
    #     data2['Fracture Spacing'] = noise_data
    if noise:
        # data1[['Treatment Pressure of Stage 1', 'Treatment Pressure of Stage 2', 'Treatment Pressure of Stage 3', 'Treatment Pressure of Stage 4', 'Treatment Pressure of Stage 5', 'Treatment Pressure of Stage 6', 'Treatment Volume of Stage 1', 'Treatment Volume of Stage 2', 'Treatment Volume of Stage 3', 'Treatment Volume of Stage 4', 'Treatment Volume of Stage 5', 'Treatment Volume of Stage 6','Fracture Spacing']] = data1[['Treatment Pressure of Stage 1', 'Treatment Pressure of Stage 2', 'Treatment Pressure of Stage 3', 'Treatment Pressure of Stage 4', 'Treatment Pressure of Stage 5', 'Treatment Pressure of Stage 6', 'Treatment Volume of Stage 1', 'Treatment Volume of Stage 2', 'Treatment Volume of Stage 3', 'Treatment Volume of Stage 4', 'Treatment Volume of Stage 5', 'Treatment Volume of Stage 6','Fracture Spacing']].apply(lambda x: (x + noise * np.std(x) * np.random.randn(x.shape[0])))
        data1[['NPV']] = data1[['NPV']].apply(lambda x: (x + noise * np.std(x) * np.random.randn(x.shape[0])))
        # noise_data = np.array(data2['Fracture Spacing'])
        # noise_data = noise * np.std(noise_data) * np.random.randn(noise_data.shape[0])
        # data2['Fracture Spacing'] = noise_datanoise * random.gauss(0, np.std(x, axis=0))
    # i = 12
    # for i in range(14):
    plt.plot(data.iloc[0:500, 13], linestyle='', marker='.')
    plt.plot(data1.iloc[0:500, 13], linestyle='', marker='*')
# plt.plot(data2.iloc[0:500, i], linestyle='', marker='.')
# 在0-5的区间上生成50个点作为测试数据
    plt.show()
