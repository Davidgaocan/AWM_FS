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

def testalldata1(datapath,r):
    import os

    output_dir = '../out'
    # data_name = os.path.splitext(datapath)[0]
    data_name = os.path.splitext(os.path.basename(datapath))[0]
    output_file_path = os.path.join(output_dir, 'tinue_wpbmfs_d_t_all_0.01', f'{data_name}.txt')

    # 检查文件是否存在
    if os.path.exists(output_file_path):
        print('文件存在：', data_name)
    print('当前遍历文件：' + data_name)
    # 获取数据
    X_train, X_test, y_train, y_test = get_scaler_data(datapath)
    d_num = X_train.shape[1]
    best_acc = 0
    best_params = {'T': 0, 'k': 0,'lamada':0,'equalvalue':0,'best_flag':0,'best_iter':0}
    best_idx = np.zeros(d_num)
    best_len = 1
    t = len(X_train) * r
    # 获取参数组合总数
    total_combinations = len([1e-3,1e-4,1e-5])\
                         *len([0.01])\
                         *len([1, 3, 5, 7, 9])
    # 使用 tqdm 显示进度条
    with tqdm(total=total_combinations, desc=f"Processing {data_name}") as pbar:
        # 遍历参数组合
        for equalvalue in [1e-3,1e-4,1e-5]:
            for lamada in [0.01]:
                for K in [1, 3, 5, 7, 9]:
                    pbar.update(1)
                    # 创建 WPBMFS 实例
                    wpbmfs = weightmaxwiththreeway_t_1(X_train, y_train, t, lamada, K, equalvalue)
                    # 获取特征权重和特征排名
                    try:
                        w,iter_t,flag= wpbmfs.fit()
                        idx = wpbmfs.feature_ranking()
                        w_final = wpbmfs.get_subset()
                    except Exception as e:
                        print("发生异常:", str(e))
                        continue
                    nonzero_count = np.count_nonzero(w_final)

                    # 训练分类器并计算准确率
                    clf = KNeighborsClassifier(n_neighbors=1)
                    clf.fit(X_train * w_final, y_train.values.ravel())
                    y_pred = clf.predict(X_test * w_final)
                    acc_weighted = accuracy_score(y_test, y_pred)

                    # 保存结果到文件
                    # output_file_path = os.path.join(output_dir, 'tinue_wpbmfs_d', f'{data_name}.txt')

                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                    with open(output_file_path, 'a') as f:
                        f.write(
                            f"T={iter_t},"
                            f"flag={flag},"
                            f"k={K},"
                            f"r={r},"
                            f"lamada={lamada},"
                            f"equalvalue={equalvalue},"
                            f"len={nonzero_count},"
                            f"acc={acc_weighted},"
                            f"idx={idx},w={w},subset={w_final},"
                            f" data_name={data_name}\n")

                    # 记录最佳准确率及对应参数和特征下标数组
                    if acc_weighted > best_acc:
                        best_acc = acc_weighted
                        best_params = {'T': iter_t, 'k': K,
                                       'lamada':lamada,
                                       'equalvalue':equalvalue,
                                       'best_flag':flag,
                                       'best_iter':iter_t}
                        best_idx = idx
                        best_subset = w_final
                        best_w = w
                        best_len = nonzero_count




        # 保存最佳准确率及对应参数和特征下标数组到文件

        output_optim_file_path = os.path.join(output_dir, 'optim_weight_d_t_all_0.01', f'{data_name}.txt')
        os.makedirs(os.path.dirname(output_optim_file_path), exist_ok=True)  # 创建文件路径
        print(
            f"Best accuracy: {best_acc}, "
            f"Best len: {best_len},"
            f"Best params: {best_params},"
            f" Best idx: {best_idx},"
            f" Best subset: {best_subset},"
            f" data_name={data_name}\n")
        with open(output_optim_file_path, 'a') as f:
            f.write(
                f"Best accuracy: {best_acc},"
                f"Best r: {r},"
                f"Best len: {best_len},"
                f"Best params: {best_params}, "
                f"Best idx: {best_idx},best_w={best_w},"
                f"best_subset={best_subset}, "
                f"data_name={data_name}\n")



