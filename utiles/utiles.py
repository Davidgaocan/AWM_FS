import logging
import multiprocessing
import sys
import time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.spatial.distance import cdist
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from algorithm.SW_simba_mutile_NH import SW_simba_plus
from algorithm.simba_plus import Simba_plus
import pandas as pd
from sklearn.mixture import GaussianMixture

def array_to_csv(path,data):
    # 构建DataFrame
    df = pd.DataFrame(data)

    # 将数据集保存为CSV文件
    df.to_csv(path, index=False,header=None)


def get_rbf_near_index():
    # 从 CSV 文件中读取数据集和标签
    df = pd.read_csv('./data/rbf_data.data')

    # 获取 label 为 1 的数据
    label_1_data = df[df['Label'] == 1][['Feature 1', 'Feature 2']]

    # 计算 label 1 数据的类中心
    label_1_center = label_1_data.mean()

    # 获取 label 为 0 的数据
    label_0_data = df[df['Label'] == 0][['Feature 1', 'Feature 2']]

    # 计算 label 0 数据与 label 1 类中心的距离
    distances = np.linalg.norm(label_0_data - label_1_center, axis=1)

    # 选择距离最近的前 100 个 label 为 0 的数据的索引
    closest_indices = label_0_data.iloc[distances.argsort()[:100]].index
    # print(closest_indices)
    # sys.exit()
    return closest_indices
    # # 绘制原始散点图
    # plt.scatter(df['Feature 1'], df['Feature 2'], c=df['Label'], cmap='viridis')
    #
    # # 在原始散点图上用圆圈标记距离最近的前 100 个 label 为 0 的数据点位置
    # plt.scatter(df.loc[closest_indices]['Feature 1'], df.loc[closest_indices]['Feature 2'], facecolors='none',
    #             edgecolors='red', marker='o', s=100)
    #
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.savefig('../img/rbf/class0_100_near.jpg')
    # plt.show()
    # sys.exit()
    # # 使用索引获取相应的数据点
    # farthest_points = label_0_data[farthest_indices]


def get_margin_and_save_mutile_file(X, Y, w_inputfile, outputfile):
    m1, n1 = np.shape(X)  # m为行数，n为列数
    margin = []


    w1 = pd.read_csv(w_inputfile[0], encoding="gbk", header=None)

    w2 = pd.read_csv(w_inputfile[1], encoding="gbk", header=None)
    w3 = pd.read_csv(w_inputfile[2], encoding="gbk", header=None)
    w4 = pd.read_csv(w_inputfile[3], encoding="gbk", header=None)


    start_time = time.time()
    for j in range(len(w1)):

        w_simba_sigmoid = w1.iloc[j]

        w_gussian = w2.iloc[j]
        w_kernel = w3.iloc[j]
        w_random= w4.iloc[j]

        al = Simba_plus(X, Y, 1, 1, 0)

        al1 = Simba_plus(X, Y, 1, 1, 0)
        al2 = Simba_plus(X, Y, 1, 1, 0)

        al3 = Simba_plus(X, Y, 1, 1, 0)

        for i in range(m1):
            al.find_nearest_neighbors_with_W(i, w_simba_sigmoid)
            al1.find_nearest_neighbors_with_W(i, w_gussian)
            al2.find_nearest_neighbors_with_W(i, w_kernel)
            al3.find_nearest_neighbors_with_W(i, w_random)

        margin1 = sum(al.margin.values())
        margin2 = sum(al1.margin.values())
        margin3 = sum(al2.margin.values())
        margin4 = sum(al3.margin.values())

        margin.append([margin1,margin2,margin3,margin4])

    array_to_csv(outputfile, margin)
    # 记录结束时间
    end_time = time.time()
    # 计算时间差
    execution_time = end_time - start_time
    print("代码执行时间：", execution_time, "秒")


def get_mean_margin_and_save_mutile_file(X, Y, w_inputfile, outputfile,k):
    m1, n1 = np.shape(X)  # m为行数，n为列数
    margin = []


    w1 = pd.read_csv(w_inputfile, encoding="gbk", header=None)

    # w2 = pd.read_csv(w_inputfile[1], encoding="gbk", header=None)

    start_time = time.time()
    for j in range(len(w1)):

        w_simba = w1.iloc[j]

        # w_random = w2.iloc[j]

        al = SW_simba_plus(X, Y, 1, 0, k)

        # al1 = Simba_plus(X, Y, 1, 1, 0)

        for i in range(m1):
            al.find_nearest_neighbors_with_W(i, w_simba)
            # al1.find_nearest_neighbors_with_W(i, w_random)
        margin1 = sum(al.margin.values())
        # margin2 = sum(al1.margin.values())

        margin.append(margin1)

    array_to_csv(outputfile, margin)
    # 记录结束时间
    end_time = time.time()
    # 计算时间差
    execution_time = end_time - start_time
    print("代码执行时间：", execution_time, "秒")

def get_rbf_far_index():
    # 从 CSV 文件中读取数据集和标签
    df = pd.read_csv('./data/rbf_data.data')

    # 获取 label 为 1 的数据
    label_1_data = df[df['Label'] == 1][['Feature 1', 'Feature 2']]

    # 计算 label 1 数据的类中心
    label_1_center = label_1_data.mean()

    # 获取 label 为 0 的数据
    label_0_data = df[df['Label'] == 0][['Feature 1', 'Feature 2']]

    # 计算 label 0 数据与 label 1 类中心的距离
    distances = np.linalg.norm(label_0_data - label_1_center, axis=1)

    # 选择距离最近的前 100 个 label 为 0 的数据的索引
    closest_indices = label_0_data.iloc[distances.argsort()[-100:]].index
    # print(closest_indices)
    # sys.exit()
    return closest_indices
    # # # 绘制原始散点图
    # plt.scatter(df['Feature 1'], df['Feature 2'], c=df['Label'], cmap='viridis')
    #
    # # # 在原始散点图上用圆圈标记距离最近的前 100 个 label 为 0 的数据点位置
    # plt.scatter(df.loc[closest_indices]['Feature 1'], df.loc[closest_indices]['Feature 2'], facecolors='none',
    #             edgecolors='red', marker='o', s=100)
    #
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.savefig('../img/rbf/class0_100_far.jpg')
    # plt.show()
    # sys.exit()
    # # 使用索引获取相应的数据点
    # farthest_points = label_0_data[farthest_indices]

def get_rbf_mid_index():
    # 从 CSV 文件中读取数据集和标签
    df = pd.read_csv('./data/rbf_data.data')

    # 获取 label 为 1 的数据
    label_1_data = df[df['Label'] == 1][['Feature 1', 'Feature 2']]

    # 计算 label 1 数据的类中心
    label_1_center = label_1_data.mean()

    # 获取 label 为 0 的数据
    label_0_data = df[df['Label'] == 0][['Feature 1', 'Feature 2']]

    # 计算 label 0 数据与 label 1 类中心的距离
    distances = np.linalg.norm(label_0_data - label_1_center, axis=1)

    # 选择距离最近的前 100 个 label 为 0 的数据的索引
    closest_indices = label_0_data.iloc[distances.argsort()[200:300]].index
    # print(closest_indices)
    # sys.exit()
    return closest_indices
    # # # 绘制原始散点图
    # plt.scatter(df['Feature 1'], df['Feature 2'], c=df['Label'], cmap='viridis')
    # #
    # # # # 在原始散点图上用圆圈标记距离最近的前 100 个 label 为 0 的数据点位置
    # plt.scatter(df.loc[closest_indices]['Feature 1'], df.loc[closest_indices]['Feature 2'], facecolors='none',
    #             edgecolors='red', marker='o', s=100)
    #
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    #
    # plt.show()
    #
    # sys.exit()
    # # 使用索引获取相应的数据点
    # farthest_points = label_0_data[farthest_indices]

def get_rbf_bound_index():
    # 从 CSV 文件中读取数据集和标签
    df = pd.read_csv('./data/rbf_data.data')

    # 获取 label 为 1 的数据
    label_1_data = df[df['Label'] == 1][['Feature 1', 'Feature 2']]

    # 计算 label 1 数据的类中心
    label_1_center = label_1_data.mean()

    # 获取 label 为 0 的数据
    label_0_data = df[df['Label'] == 0][['Feature 1', 'Feature 2']]
    label_0_center = label_0_data.mean()

    # 计算 label 0 数据与 label 1 类中心的距离
    distances = np.linalg.norm(label_0_data - label_0_center, axis=1)

    # 选择距离最近的前 100 个 label 为 0 的数据的索引
    closest_indices = label_0_data.iloc[distances.argsort()[-100:]].index
    # print(closest_indices)
    # sys.exit()
    return closest_indices
    # # # # 绘制原始散点图
    # plt.scatter(df['Feature 1'], df['Feature 2'], c=df['Label'], cmap='viridis')
    # #
    # # # # 在原始散点图上用圆圈标记距离最近的前 100 个 label 为 0 的数据点位置
    # plt.scatter(df.loc[closest_indices]['Feature 1'], df.loc[closest_indices]['Feature 2'], facecolors='none',
    #             edgecolors='red', marker='o', s=100)
    #
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    #
    # plt.show()
    # sys.exit()
    # # 使用索引获取相应的数据点
    # farthest_points = label_0_data[farthest_indices]


# Function to find the k nearest indices, modified to handle three-dimensional data
def find_k_nearest_indices(x, y, k):
    unique_labels = np.unique(y)
    nearest_indices = []
    far_indices = []
    mid_indices = []

    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        label_data = x[label_indices]
        other_data = x[y != label]
        sample_num=int((1/3)*len(label_indices))
        # Compute distance between each sample in the label_data and the nearest sample in other_data
        distances = cdist(label_data, other_data, metric='euclidean')
        min_distances = np.min(distances, axis=1)

        # Get the indices of the k smallest distances
        k_nearest_indices = label_indices[np.argsort(min_distances)[:sample_num]]
        k_far_indices = label_indices[np.argsort(min_distances)[sample_num+1:sample_num*2]]
        k_mid_indices = label_indices[np.argsort(min_distances)[-sample_num:]]

        nearest_indices.extend(k_nearest_indices)
        far_indices.extend(k_far_indices)
        mid_indices.extend(k_mid_indices)

    return nearest_indices, far_indices, mid_indices


def save_margin_plot(path,imgfile):
    # 从 CSV 文件中读取数据
    data = pd.read_csv(path, header=None)

    # 获取 Simba 实验结果、Relief 实验结果和正确特征向量结果的列数据
    simba_results = data
    # relief_results = data['feature_2']
    # correct_results = data['feature_3']
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimSun']
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()

    # 设置横坐标和纵坐标的刻度间隔
    x_ticks = range(0, len(data), int(len(data) / 10))
    y_ticks = range(0, int(np.max(data)[0]) + 1, int((np.max(data)[0]) / 10))

    # 绘制 Simba 实验结果的折线图（粗实线）
    ax.plot(simba_results, linestyle='-', linewidth=2, label='Simba')

    # # 绘制 Relief 实验结果的折线图（细虚线）
    # ax.plot(relief_results, linestyle=':', linewidth=1, label='Relief')

    # # 绘制正确特征向量结果的折线图（粗虚线）
    # ax.plot(correct_results, linestyle='--', linewidth=2, label='Correct')

    # 设置图例
    ax.legend()

    # 设置横坐标和纵坐标的刻度
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # 设置横坐标和纵坐标的标签
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Margin')

    # 显示图形
    plt.savefig(imgfile)
    plt.show()
    plt.close()


def save_two_margin_plot(path,imgfile):
    # 从 CSV 文件中读取数据
    data = pd.read_csv(path, header=None)

    # 获取 Simba 实验结果、Relief 实验结果和正确特征向量结果的列数据

    sigmoid_results = data[0]
    gaussian_results = data[1]
    kerneldensity_results = data[2]
    random_results = data[3]

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimSun']
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()

    # 设置横坐标和纵坐标的刻度间隔
    # x_ticks = np.linspace(0, len(data), 10)
    #
    # y_ticks = np.linspace((np.min(data)[0]), (np.max(data)[0]) + 1,10)

    # # 绘制 Simba 实验结果的折线图（粗实线）
    # ax.plot(simba_results, linestyle='-', linewidth=2, label='Simba')

    # 绘制 Relief 实验结果的折线图（细虚线）
    ax.plot(sigmoid_results, linestyle='-', linewidth=2, label='Sample_Weight_sigmoid')

    ax.plot(gaussian_results, linestyle='-', linewidth=2, label='Sample_Weight_gaussian')

    ax.plot(kerneldensity_results, linestyle='-', linewidth=2, label='Sample_Weight_kernel')

    # 绘制正确特征向量结果的折线图（粗虚线）
    ax.plot(random_results, linestyle='-',linewidth=2, label='Random')

    # 设置图例
    ax.legend()

    # 设置横坐标和纵坐标的刻度
    # ax.set_xticks(x_ticks)
    # ax.set_yticks(y_ticks)

    # 设置横坐标和纵坐标的标签
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Margin')

    # 显示图形
    plt.savefig(imgfile)
    plt.show()
    plt.close()

def save_mutile_margin_plot(path,imgfile):
    # 从 CSV 文件中读取数据
    data1 = pd.read_csv(path[0], header=None)
    data = pd.read_csv(path[1], header=None)
    data2 = pd.read_csv(path[2], header=None)

    # 获取 Simba 实验结果、Relief 实验结果和正确特征向量结果的列数据

    weight_results = data[0]
    random_results = data[1]

    swsimba = data1[0]

    swsimba_plus = data2[0]


    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimSun']
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()

    # 设置横坐标和纵坐标的刻度间隔
    x_ticks = np.linspace(0, len(data), 10)

    y_ticks = np.linspace((np.min(data)[0]), (np.max(data)[0]) + 1,10)

    # # 绘制 Simba 实验结果的折线图（粗实线）
    # ax.plot(simba_results, linestyle='-', linewidth=2, label='Simba')

    # 绘制 Relief 实验结果的折线图（细虚线）
    ax.plot(weight_results, linestyle=':', linewidth=1, label='Sample_Weight')

    # 绘制正确特征向量结果的折线图（粗虚线）
    ax.plot(random_results, linestyle='--', linewidth=2, label='Random')

    # 绘制正确特征向量结果的折线图（粗虚线）
    ax.plot(swsimba, linestyle='-', linewidth=2, label='swsimba')

    ax.plot(swsimba_plus, linestyle='-.', linewidth=2, label='swsimba_plus')

    # 设置图例
    ax.legend()

    # 设置横坐标和纵坐标的刻度
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # 设置横坐标和纵坐标的标签
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Margin')

    # 显示图形
    plt.savefig(imgfile)
    plt.show()
    plt.close()

'''
根据path获取到标准化的数据
'''
def get_scaler_data(data_path):
    df = pd.read_csv(data_path, header=None)
    X = df.iloc[:, :-1]

    # 创建一个标准化缩放器
    scaler = StandardScaler()

    # 对数据进行标准化
    X = scaler.fit_transform(X)

    # 将缩放后的数据重新转换为DataFrame
    X = pd.DataFrame(X)
    Y = df.iloc[:, -1]



    return X,Y

def get_data(data_path):
    df = pd.read_csv(data_path, header=None)
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]

    return X,Y

def get_margin(X,Y,w_inputfile,outputfile):

    al = Simba_plus(X, Y, 1, 1, 0)
    m1, n1 = np.shape(X)  # m为行数，n为列数
    margin = []
    w1 = pd.read_csv(w_inputfile, encoding="gbk",header=None)
    start_time = time.time()
    for j in range(m1):
        w_simba=w1.iloc[j]
        # w_relief=w2.iloc[j]
        al = Simba_plus(X, Y, 1, 1, 0)
        # al1 = Simba_plus(X, Y, 0.1, 1, 0)
        for i in range(m1):
            al.find_nearest_neighbors_with_W(i, w_simba)
            # al1.find_nearest_neighbors_with_W(i, w_relief)
        margin1=sum(al.margin.values())
        # margin2=sum(al1.margin.values())
        margin.append([margin1])

    array_to_csv(outputfile, margin)
    # 记录结束时间
    end_time = time.time()
    # 计算时间差
    execution_time = end_time - start_time
    print("代码执行时间：", execution_time, "秒")




'''
获取near数据、mid数据下标、以及far数据下标
1.获取到三类样本点：dmax是指最大异类距离，探究哪个部分点的结果特征权重更接近于实验整体得出的特征权重：
near：[0,dmax*0.25]之间的点
mid：[dmax*0.25,dmax*0.75]
far:[>dmax*0.75]
'''
def find_t_nearest_indices(x, y, t):
    x=x.values
    y=y.values
    unique_labels = np.unique(y)

    nearest_indices = []
    far_indices = []
    mid_indices = []
    near_far=[]
    near_mid=[]
    far_mid=[]

    for label in unique_labels:
        label_indices = np.where(y == label)[0]

        label_data = x[label_indices]
        other_data = x[y != label]
        sample_num=int((1/3)*len(label_indices))
        # Compute distance between each sample in the label_data and the nearest sample in other_data
        distances = cdist(label_data, other_data, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        maxdis=np.max(min_distances)
        neardis=maxdis*t
        fardis=maxdis*(1-t)

        # Get the indices of the k smallest distances
        # k_nearest_indices = label_indices[np.argsort(min_distances)[:sample_num]]
        k_nearest_indices = label_indices[np.where(min_distances<neardis)]


        k_far_indices = label_indices[np.where(min_distances>fardis)]
        k_mid_indices = label_indices[np.where((min_distances<=fardis)&(min_distances>=neardis))]
        near_far.extend(k_far_indices)
        near_far.extend(k_nearest_indices)

        near_mid.extend(k_mid_indices)
        near_mid.extend(k_nearest_indices)

        far_mid.extend(k_far_indices)
        far_mid.extend(k_mid_indices)

        # print(k_nearest_indices)
        # print(k_far_indices)
        # print(k_mid_indices)
        # print(np.any(np.in1d(near_mid, k_mid_indices)))
        # sys.exit()

        nearest_indices.extend(k_nearest_indices)
        far_indices.extend(k_far_indices)
        mid_indices.extend(k_mid_indices)

    return nearest_indices, far_indices, mid_indices,near_mid,near_far,far_mid

import random


import random

def generate_random_weights(num, nearhit_indices,k):
    indices=list(range(num))
    # 生成随机权重
    weights = []
    for idx in indices:
        if idx in nearhit_indices[0]:
            weight = random.uniform(0.75, 1)
        elif idx in nearhit_indices[1]:
            weight = random.uniform(0.25, 0.5)
        elif idx in nearhit_indices[2]:
            weight = random.uniform(0, 0.25)
        else:
            weight = 1.0
        weights.append(weight)

    # 根据权重进行随机抽样
    random_sample = random.choices(indices, weights=weights, k=k)

    # 获取对应下标的权重值列表
    weights_of_samples = [weights[idx] for idx in random_sample]
    # print((random_sample))
    #
    # print(weights_of_samples)
    # sys.exit()
    return random_sample, weights

def get_sample_with_weigth(weight,num,k):
    indices=list(range(num))

    # 根据权重进行随机抽样
    random_sample = random.choices(indices, weights=weight, k=k)

    return random_sample
#超时
def run_with_timeout(target,timeout):
    process = multiprocessing.Process(target=target)
    process.start()
    process.join(timeout)

    if process.is_alive():
        print(f"Task exceeded the specified timeout of {timeout} seconds. Stopping the task.")
        process.terminate()
        process.join()

def z_score_normalize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    normalized_arr = (arr - mean) / std
    return normalized_arr

def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr


#高斯分布拟合出输出概率
def GaussianMixtureToProb(path):
    import pandas as pd
    import numpy as np
    from scipy.stats import multivariate_normal

    # 读取CSV文件
    data = pd.read_csv(path)

    # 提取特征数据和标签数据
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    # 获取所有不重复的标签类别
    unique_labels = labels.unique()

    # 用于存储每个类别的高斯分布模型
    gaussian_models = {}

    # 对于每个标签类别，拟合多变量高斯分布模型
    for label in unique_labels:
        label_data = features[labels == label]
        mean = label_data.mean()
        cov = label_data.cov()
        gaussian_models[label] = multivariate_normal(mean=mean, cov=cov)

    # 遍历每个样本点，计算其在对应类别的高斯分布下的概率值
    probabilities = []

    for index, row in features.iterrows():

        label=labels.iloc[index]
        gaussian_model = gaussian_models[label]

        probability = gaussian_model.pdf(row)
        probabilities.append(probability)


    # 打印每个样本点在不同类别下的概率值
    # print(probabilities)
    # print(max(probabilities))
    # print(min(probabilities))

    probabilities=min_max_normalize(probabilities)

    # print(probabilities)
    # print(max(probabilities))
    # print(min(probabilities))
    return probabilities

def GaussianMixtureToProb_plus(X,Y):
    import pandas as pd
    import numpy as np
    from scipy.stats import multivariate_normal
    from sklearn.covariance import LedoitWolf


    # 提取特征数据和标签数据
    features = X
    labels = Y

    # 获取所有不重复的标签类别
    unique_labels = labels.unique()

    # 用于存储每个类别的高斯分布模型
    gaussian_models = {}

    # 对于每个标签类别，拟合多变量高斯分布模型
    for label in unique_labels:
        label_data = features[labels == label]
        mean = label_data.mean()
        # cov = label_data.cov()
        # 使用 Ledoit-Wolf 估计方法计算稳定的协方差矩阵
        lw = LedoitWolf()
        cov = lw.fit(label_data).covariance_

        gaussian_models[label] = multivariate_normal(mean=mean, cov=cov)

    # 遍历每个样本点，计算其在对应类别的高斯分布下的概率值
    probabilities = []

    for index, row in features.iterrows():

        label=labels.iloc[index]
        gaussian_model = gaussian_models[label]

        probability = gaussian_model.pdf(row)
        probabilities.append(probability)

    # print(max((DataFrame(probabilities)[labels==1]).values)[0])
    # print()
    # print(unique_labels)
    max_value={i:max((DataFrame(probabilities)[labels==i]).values)[0] for i in unique_labels}

    min_value={i:min(((DataFrame(probabilities)[labels==i]).values))[0] for i in unique_labels}
    # print(probabilities)
    probabilities_new=[round((probabilities[i]-min_value[labels[i]])/(max_value[labels[i]]-min_value[labels[i]]),2) for i in range(len(probabilities))]
    probabilities=[1-i for i in probabilities_new]

    high_prob_indices = np.where(DataFrame(probabilities) < 0.7)[0]
    # print(high_prob_indices)
    # # 绘制概率较高的数据分布
    #
    # plt.scatter(features.iloc[:, 0], features.iloc[:, 1],
    #             label='Low Probability')
    # plt.scatter(features.iloc[high_prob_indices, 0], features.iloc[high_prob_indices, 1], color='red',
    #             label='High Probability')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.title('Data Distribution with High Probability')
    # plt.legend()
    # plt.show()
    # # 设置中文字体
    # plt.rcParams['font.sans-serif'] = ['SimSun']
    # # 创建图形和坐标轴对象
    # fig, ax = plt.subplots()
    # ax.scatter(features.iloc[:, 0], features.iloc[:, 1],c=labels.iloc[:]
    #            )
    # ax.scatter(features.iloc[high_prob_indices, 0], features.iloc[high_prob_indices, 1]
    #            , color='red', label='High Probability')
    # # ax.scatter(features.iloc[~high_prob_indices, 0], features.iloc[~high_prob_indices, 1],
    # #            features.iloc[~high_prob_indices, 2], color='red', label='Low Probability')
    # print(len(features))
    #
    # ax.set_xlabel('Feature 1')
    # ax.set_ylabel('Feature 2')
    #
    # ax.set_title('Data Distribution with High Probability')
    # plt.legend()
    # plt.show()
    # print(max(probabilities))
    # print(min(probabilities))
    # 打印每个样本点在不同类别下的概率值
    # print(probabilities)
    # print(max(probabilities))
    # print(min(probabilities))

    # probabilities=min_max_normalize(probabilities)

    # print(probabilities)
    # print(max(probabilities))
    # print(min(probabilities))
    return probabilities

def kerneldensity(x,y):
    import pandas as pd
    import numpy as np
    from sklearn.neighbors import KernelDensity

    # 提取特征数据和标签数据
    features = x
    labels = y

    # 获取所有不重复的标签类别
    unique_labels = labels.unique()

    # 用于存储每个类别的核密度估计模型
    kde_models = {}

    # 对每个标签类别，拟合核密度估计模型
    for label in unique_labels:
        label_data = features[labels == label]
        kde_model = KernelDensity(bandwidth=0.2)  # 可调整带宽参数
        kde_model.fit(label_data)
        kde_models[label] = kde_model

    probabilities = []

    for index, row in features.iterrows():
        label = labels.iloc[index]
        chosen_kde_model = kde_models[label]
        log_density = chosen_kde_model.score_samples(row.values.reshape(1, -1))

        # 判断是否溢出
        probability = np.exp(log_density)

        probabilities.append(probability[0])



    # 选择概率较高的数据点的索引
    # 将概率值转换为numpy数组
    probabilities = np.array(probabilities)
    # print(probabilities)


    max_value = {i: max((DataFrame(probabilities)[labels == i]).values)[0] for i in unique_labels}

    min_value = {i: min(((DataFrame(probabilities)[labels == i]).values))[0] for i in unique_labels}
    # print(probabilities)
    probabilities_new = [
        round((probabilities[i] - min_value[labels[i]]) / (max_value[labels[i]] - min_value[labels[i]]), 2)
         if max_value[labels[i]] != min_value[labels[i]] else 0
         for i in range(len(probabilities))  ]

    probabilities = [1 - i for i in probabilities_new]
    # print(probabilities)
    # sys.exit()
    # high_prob_indices = np.where(DataFrame(probabilities) > 0.9)[0]
    #
    # # 设置中文字体
    # plt.rcParams['font.sans-serif'] = ['SimSun']
    # # 创建图形和坐标轴对象
    # fig, ax = plt.subplots()
    # ax.scatter(features.iloc[:, 0], features.iloc[:, 1],c=labels.iloc[:]
    #            )
    # ax.scatter(features.iloc[high_prob_indices, 0], features.iloc[high_prob_indices, 1]
    #            , color='red', label='High Probability')
    # # ax.scatter(features.iloc[~high_prob_indices, 0], features.iloc[~high_prob_indices, 1],
    # #            features.iloc[~high_prob_indices, 2], color='red', label='Low Probability')
    # print(len(features))
    #
    # ax.set_xlabel('Feature 1')
    # ax.set_ylabel('Feature 2')
    #
    # ax.set_title('Data Distribution with High Probability')
    # plt.legend()
    # plt.show()

    # print(high_prob_indices)
    # # 绘制概率较高的数据分布
    #
    # plt.scatter(features.iloc[:, 0], features.iloc[:, 1],
    #             label='Low Probability')
    # plt.scatter(features.iloc[high_prob_indices, 0], features.iloc[high_prob_indices, 1], color='red',
    #             label='High Probability')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.title('Data Distribution with High Probability')
    # plt.legend()
    # plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(features.iloc[:, 0], features.iloc[:, 1],
    #            features.iloc[:, 2], label='Low Probability')
    # ax.scatter(features.iloc[high_prob_indices, 0], features.iloc[high_prob_indices, 1],
    #            features.iloc[high_prob_indices, 2],color='red', label='High Probability')
    # # ax.scatter(features.iloc[~high_prob_indices, 0], features.iloc[~high_prob_indices, 1],
    # #            features.iloc[~high_prob_indices, 2], color='red', label='Low Probability')
    # print(len(features))
    #
    # ax.set_xlabel('Feature 1')
    # ax.set_ylabel('Feature 2')
    # ax.set_zlabel('feature 3')
    # ax.set_title('Data Distribution with High Probability')
    # plt.legend()
    # plt.show()
    return probabilities


def sigmoid_function(z, beta):
    return 1 / (1 + np.exp(beta * z))

#计算margin小的概率大
def margin_weight(beta,margin):
    return sigmoid_function(margin,beta)


def find_max_gap_with_value(arr):
    # 首先对数组进行排序
    arr.sort()

    max_gap = 0
    smaller_value = None

    # 遍历排序后的数组，计算相邻元素之间的间隙，并记录较小的值
    for i in range(1, len(arr)):
        gap = arr[i] - arr[i - 1]
        if gap > max_gap:
            max_gap = gap
            smaller_value = arr[i - 1]

    return smaller_value


def mid_subset_w(w_init):

    # 计算数组的平均值 t
    t = sum(w_init) / len(w_init)

    # 使用列表推导来将数组中的每个元素根据 t 转换为 1 或 0
    result = [1 if x > t else 0 for x in w_init]

    # 打印结果
    return result



def getLogfile(inputfile):
    # 配置日志
    logging.basicConfig(filename=inputfile, level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 创建控制台处理程序并设置其级别
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 控制台只显示INFO级别及以上的日志信息

    # 创建格式化器并将其添加到控制台处理程序
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)

    # 将控制台处理程序添加到日志记录器
    logging.getLogger('').addHandler(console_handler)

    # 重定向print输出到日志文件
    class LoggerWriter:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level

        def write(self, message):
            if message != '\n':
                self.logger.log(self.level, message)

        def flush(self):
            pass

    # 将stdout重定向到日志文件
    sys.stdout = LoggerWriter(logging.getLogger('stdout'), logging.INFO)

def max_k_elements_to_ones(w, k):
    # 获取数组w中前k个最大的元素的索引
    max_indices = sorted(range(len(w)), key=lambda i: w[i], reverse=True)[:k]

    # 创建新数组，并将最大的前k个元素位置设为1，其余位置设为0
    result_array = [1 if i in max_indices else 0 for i in range(len(w))]

    return result_array


def get_scaler_data(data_path):
    df = pd.read_csv(data_path, header=None)
    X = df.iloc[:, :-1]

    # 创建一个标准化缩放器
    scaler = StandardScaler()

    # 对数据进行标准化
    X = scaler.fit_transform(X)

    # 将缩放后的数据重新转换为DataFrame
    X = pd.DataFrame(X)
    Y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test







