import argparse
import sys

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.svm import SVC
from algorithm.weightmax import weightmax
from algorithm.weightmaxwiththreeway import weightmaxwiththreeway
from algorithm.weightmaxwiththreeway_t import weightmaxwiththreeway_t
from algorithm.weightmaxwiththreeway_t_1 import weightmaxwiththreeway_t_1
from sklearn.svm import LinearSVC


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
    # 重置索引，保持列名不变
    # 将 Series 转换回 DataFrame 类型，并重置列索引
    Y=pd.DataFrame(Y)
    Y.columns=[0]

    return X,Y

def find_nonzero_indices(arr):
    indices = [index for index, value in enumerate(arr) if value != 0]
    return indices

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
    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(features), pd.DataFrame(labels), test_size=0.2, random_state=42)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test

def testalldata(datapath,r,daname,lam):
    import os

    output_dir = '../outtop_svm_sigmoid'
    # data_name = os.path.splitext(datapath)[0]
    data_name = os.path.splitext(os.path.basename(datapath))[0]
    output_file_path = os.path.join(output_dir, f'tinue_our_{daname}_{lam}', f'{data_name}.txt')

    # 检查文件是否存在
    if os.path.exists(output_file_path):
        print('文件存在：', data_name)
        return
    print('当前遍历文件：' + data_name)
    # 获取数据
    X, Y = get_scaler_data(datapath)
    Y_copy = np.array(Y).T[0]

    X_copy = np.array(X)

    d_num = X.shape[1]
    best_acc = 0
    best_params = {'T': 0, 'k': 0,'lamada':0,'equalvalue':0,'best_flag':0,'best_iter':0}
    best_idx = np.zeros(d_num)
    best_len = 1
    t = len(X) * r if len(X) * r<10000 else len(X)
    # 获取参数组合总数
    total_combinations = len([1e-3,1e-4,1e-5])\
                         *len([lam])\
                         *len([1, 3, 5, 7, 9])
    # 使用 tqdm 显示进度条
    with tqdm(total=total_combinations, desc=f"Processing {data_name}") as pbar:
        # 遍历参数组合
        for equalvalue in [1e-3,1e-4,1e-5]:
            for lamada in [lam]:
                for K in [1, 3, 5, 7, 9]:
                    pbar.update(1)
                    # 创建 WPBMFS 实例
                    wpbmfs = weightmaxwiththreeway_t_1(X, Y, t, lamada, K, equalvalue)

                    # 获取特征权重和特征排名
                    try:
                        w,iter_t,flag= wpbmfs.fit()
                        idx = wpbmfs.feature_ranking()
                        w_final = wpbmfs.get_subset()
                    except Exception as e:
                        print("发生异常:", str(e))
                        continue
                    nonzero_count = np.count_nonzero(w_final)
                    ss = model_selection.KFold(n_splits=10, shuffle=True,random_state=42)
                    # clf = KNeighborsClassifier(n_neighbors=1)
                    clf =  SVC(kernel='sigmoid', random_state=42)
                    correct = 0



                    index=find_nonzero_indices(w_final)

                    for train, test in ss.split(X):
                        selected_features = X_copy[:, index]

                        clf.fit(selected_features[train], Y_copy[train])
                        # predict the class labels of test data
                        y_predict = clf.predict(selected_features[test])
                        acc = accuracy_score(Y_copy[test], y_predict)
                        correct = correct + acc

                    acc_val = float(correct) / 10
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
                            f"acc={acc_val},"
                            f"idx={idx},w={w},subset={w_final},"
                            f" data_name={data_name}\n")

                    # 记录最佳准确率及对应参数和特征下标数组
                    if acc_val >= best_acc:
                        best_acc = acc_val
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

        output_optim_file_path = os.path.join(output_dir, f'optim_our_{daname}_{lam}', f'{data_name}.txt')
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
        output_optim_file_path_our = os.path.join(output_dir, f'subset_our_{daname}_{lam}', f'{data_name}.txt')
        os.makedirs(os.path.dirname(output_optim_file_path_our), exist_ok=True)
        with open(output_optim_file_path_our, 'a') as f:
            f.write(
                f"subset={best_subset} ")





