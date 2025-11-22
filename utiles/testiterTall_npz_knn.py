import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from algorithm.weightmax import weightmax
from algorithm.weightmaxwiththreeway import weightmaxwiththreeway
from algorithm.weightmaxwiththreeway_t import weightmaxwiththreeway_t
from algorithm.weightmaxwiththreeway_t_1 import weightmaxwiththreeway_t_1


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

def get_scaler_data1(data_path):
    # load dataset
    dataset = np.load(data_path)

    # obtain dataset's features and labels
    features, labels, is_dis, missing = (
        dataset['features'],
        dataset['labels'],
        dataset['is_dis'],
        dataset['missing']
    )
    # 创建一个标准化缩放器
    scaler = StandardScaler()
    # 对数据进行标准化
    features = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(features), pd.DataFrame(labels).iloc[:,-1], test_size=0.2, random_state=42)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test

def testalldata(datapath,daname):
    import os

    output_dir = '../out'
    # data_name = os.path.splitext(datapath)[0]
    data_name = os.path.splitext(os.path.basename(datapath))[0]
    output_file_path = os.path.join(output_dir, f'tinue_wpbmfs_d_t_{daname}', f'{data_name}.txt')

    # 检查文件是否存在
    if os.path.exists(output_file_path):
        print('文件存在：', data_name)
        return
    print('当前遍历文件：' + data_name)
    # 获取数据
    X_train, X_test, y_train, y_test = get_scaler_data1(datapath)

    d_num = X_train.shape[1]
    # 训练分类器并计算准确率
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train , y_train.values.ravel())
    y_pred = clf.predict(X_test)
    acc_weighted = accuracy_score(y_test, y_pred)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'a') as f:
        f.write(
            f"acc={acc_weighted},"
            f" data_name={data_name}\n")













