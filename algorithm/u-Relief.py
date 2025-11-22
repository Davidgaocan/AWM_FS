import math
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler


def diff(A, near, X, index):
    maxf = X[:, A].max()
    minf = X[:, A].min()
    return math.pow(np.round(abs(X[index, A] - X[near, A]) / (maxf - minf), 2), 2)
    # return math.pow(round(abs(X[index, A] - X[near, A]), 2), 2)



def md(X, xi, xr):
    sum = 0
    for i in range(X.shape[1]):
        sum += np.abs(X[xi][i] - X[xr][i])
    return sum
def mean_Relief(X, Y, k):
    print("uRelief")
    N = X.shape[1]
    m = X.shape[0]
    W = np.zeros((N, 1))
    labelset = np.unique(Y)
    # 创建一个空字典用于存放不同类别的索引
    class_indices = {}
    # 遍历标签列表 Y，将索引归类
    for idx, label in enumerate(Y):
        if label not in class_indices:
            class_indices[label] = [idx]
        else:
            class_indices[label].append(idx)

    # # 打印结果
    # for class_label, indices in class_indices.items():
    #     print(f'Class {class_label} Indices: {indices}')
    P = {class_label: [] for class_label in labelset}
    for class_label, indices in class_indices.items():
        P[class_label] = len(indices) / X.shape[0]
    i = 0
    while i < m:
        index = random.randint(0, X.shape[0] - 1)
        D = {class_label: [] for class_label in labelset}
        for class_label, indices in class_indices.items():
            for xr in indices:
                if class_label not in D:
                    D[class_label] = md(X, index, xr)
                else:
                    D[class_label].append(md(X, index, xr))
        avg = {class_label: [] for class_label in labelset}
        for class_label, indices in D.items():
            sum_value = 0
            for xr in indices:
                sum_value += xr
            sum_value /= len(indices)
            avg[class_label] = sum_value
        D_star = {class_label: [] for class_label in labelset}
        for class_label, indices in D.items():
            for xr in indices:
                D_star[class_label].append(xr - avg[class_label])
        far_K = {class_label: [] for class_label in labelset}
        # 遍历字典，对每个类别的值进行从大到小排序并获取排序后的索引
        for class_label, values in D_star.items():
            # 使用 enumerate 获取原始索引和数值，然后按数值从大到小排序
            sorted_indices = sorted(enumerate(values), key=lambda x: x[1], reverse=True)
            # 提取排序后的索引
            sorted_indices = [index for index, _ in sorted_indices]
            for j in range(k):
                far_K[class_label].append(class_indices[class_label][sorted_indices[j]])
        a = 0
        while a < N:
            dh = 0
            dm = 0
            for j in range(k):
                dh += diff(a, far_K[Y[index]][j], X, index)
            dh = dh * (m * k)
            for c, values in far_K.items():
                if c == Y[index]:
                    continue
                PP = P[c] / 1 - P[Y[index]]
                sum = 0
                for j in range(k):
                    sum += diff(a, far_K[c][j], X, index)
                dh += PP * sum
            dh = dh * (m * k)
            W[a] = W[a] - dh + dm
            a += 1
        i += 1
    W = W / float(m)
    scale = MinMaxScaler()
    W = scale.fit_transform(W)
    return W