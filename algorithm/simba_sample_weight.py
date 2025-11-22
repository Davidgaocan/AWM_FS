
import math
import sys
from collections import defaultdict
import time

import pandas as pd
import numpy as np
import numpy.linalg as la
import random
import csv

from sklearn.preprocessing import LabelEncoder

from algorithm.Nsga2_simba import NSGA2_simba
from utiles import utiles

'''
适用于多分类问题
'''
class Simba_sample_weight:
    def __init__(self, X,Y, sample_rate, t, k):

        self.X = X

        label_encoder = LabelEncoder()
        # self.Y = label_encoder.fit_transform(Y)
        self.Y=Y

        self.X=self.process_discrete_features(self.X)


        self.__sample_num = int(round(len(X) * sample_rate))
        self.__t = t
        self.__k = k

        self.W=[1]*((X.shape[1]))

        self.nearhit={}
        self.nearhitdis={}
        self.nearmissdis={}
        self.nearmiss={}
        self.margin={}
        self.e={}
        self.nearhit_feature_dis = defaultdict(dict)
        self.nearmiss_feature_dis= defaultdict(dict)
    # 数据处理（将离散型数据处理成连续型数据，比如字符到数值）
    def process_discrete_features(selif,x):
        new_dataframe = x.copy()
        label_encoder = LabelEncoder()
        for column in new_dataframe.columns:
            if new_dataframe[column].dtype == 'object':
                new_dataframe[column] = label_encoder.fit_transform(new_dataframe[column])

        return new_dataframe


    def find_nearest_neighbors(self, i):

        X = self.X
        Y = self.Y
        weights = self.W

        # Calculate the weighted differences for all samples at once
        weighted_diff = np.linalg.norm(weights * (X - X.iloc[i]), axis=1)

        same_class_indices = np.where(Y == Y.iloc[i])[0]

        same_class_indices = same_class_indices[same_class_indices != i]

        different_class_indices = np.where(Y != Y.iloc[i])[0]

        argmin_near = np.argmin(weighted_diff[same_class_indices])
        nearest_same_class_sample_index = same_class_indices[argmin_near]

        nearest_same_class_sample_distance = weighted_diff[same_class_indices][argmin_near]

        argmin_diff = np.argmin(weighted_diff[different_class_indices])
        nearest_different_class_sample_index = different_class_indices[argmin_diff]
        nearest_different_class_sample_distance = weighted_diff[different_class_indices][argmin_diff]


        self.nearhit[i] = nearest_same_class_sample_index

        self.nearmiss[i] = nearest_different_class_sample_index

        self.nearhitdis[i] = round(nearest_same_class_sample_distance, 2)
        self.nearmissdis[i] = round(nearest_different_class_sample_distance, 2)

        self.margin[i] = (self.nearmissdis[i] - self.nearhitdis[i]) * (1 / 2)

        return nearest_same_class_sample_index, nearest_different_class_sample_index



    def find_nearest_neighbors_with_W(self, i,W):
        # 记录开始时间
        # start_time = time.time()
        X=self.X
        Y=self.Y
        self.W=W
        weights=W


        # Calculate the weighted differences for all samples at once
        weighted_diff = np.linalg.norm(weights * (X - X.iloc[i]), axis=1)



        # Find the nearest same class and different class samples
        same_class_indices = np.where(Y == Y.iloc[i])[0]

        same_class_indices = same_class_indices[same_class_indices != i]  # Remove the index i from the same class indices

        different_class_indices = np.where(Y != Y.iloc[i])[0]
        # print(same_class_indices)
        argmin_near=np.argmin(weighted_diff[same_class_indices])
        nearest_same_class_sample_index = same_class_indices[argmin_near]


        nearest_same_class_sample_distance = weighted_diff[same_class_indices][argmin_near]

        argmin_diff=np.argmin(weighted_diff[different_class_indices])
        nearest_different_class_sample_index = different_class_indices[argmin_diff]
        nearest_different_class_sample_distance = weighted_diff[different_class_indices][argmin_diff]



        # sys.exit()
        # end_time = time.time()

        # 计算程序运行时间
        # runtime = end_time - start_time
        #
        # print(f"程序运行时间：{runtime}秒")
        # print(nearest_same_class_sample_index)
        # print(nearest_different_class_sample_index)
        # sys.exit()

        self.nearhit[i]=nearest_same_class_sample_index

        self.nearmiss[i]=nearest_different_class_sample_index

        self.nearhitdis[i]=round(nearest_same_class_sample_distance,2)
        self.nearmissdis[i]=round(nearest_different_class_sample_distance,2)

        self.margin[i]=(self.nearmissdis[i]-self.nearhitdis[i])*(1/2)

        return nearest_same_class_sample_index, nearest_different_class_sample_index


    def get_weight(self, feature, index):

        data = self.X
        row = data.iloc[index]

        nearhit = data.iloc[self.nearhit[index]]
        right_w = pow(round(abs(row[feature] - nearhit[feature]), 2), 2)

        nearmiss = data.iloc[self.nearmiss[index]]

        wrong_w = pow(round(abs(row[feature] - nearmiss[feature]), 2), 2)

        fronter = 0 if (self.nearmissdis[index] == 0) else (wrong_w / self.nearmissdis[index])

        laster = 0 if (self.nearhitdis[index] == 0) else (right_w / self.nearhitdis[index])

        self.e[index]=round(fronter - laster,2)*(1/2)


        self.nearhit_feature_dis[index][feature]=right_w

        self.nearmiss_feature_dis[index][feature]=wrong_w
        return right_w,wrong_w

    # 过滤式特征选择
    def reliefF(self,outputfile='../out/xor/xor_w.csv',sample_way_isSampleWeight=True):

        # print sample
        m, n = np.shape(self.X)  # m为行数，n为列数

        data_w = []
        data_w.append([1]*n)

        #判断是否是按照样本权重采样
        if sample_way_isSampleWeight:
            # 获取到每个位置上的坐标：
            nearest_indices, far_indices, mid_indices, near_mid, near_far, far_mid = utiles.find_t_nearest_indices(
                self.X, self.Y, 0.25)

            # 获取到near、mid、far下标值
            index = [nearest_indices, mid_indices, far_indices]
            # 如果按照样本权重采样，则通过赋予不同位置上的权重。然后按照权重重复采样
            sample_index,weight = utiles.generate_random_weights(m,index,self.__sample_num)[0]
            # self.weight=(weight[i] for i in index)
        else:
            #否则按照随机取样方法进行
            # print(self.__sample_num)
            # print(m)
            sample_index=random.sample(range(0, m), self.__sample_num)



        for i in sample_index:  # 采样次数

            try:
                NearHit, NearMiss = self.find_nearest_neighbors(i)
            except:
                continue

            # print('第 %s 次采样，样本index为 %s，其NearHit k近邻行索引为 %s ，NearMiss k近邻行索引为 %s' % (num, i, NearHit, NearMiss))
            for f in self.X.columns:
                # print('***:', f, i, NearHit, NearMiss)

                self.get_weight(f, i)

                f_index = self.X.columns.get_loc(f)
                w_range = round(self.e[i] * (self.W[f_index]), 2)
                self.W[f_index] += w_range

                # print('特征 %s 的权重变化为 %s.' % (f, w_range))
            # score.append(one_score)

            data_w.append(self.get_norm(self.W.copy()))

        utiles.array_to_csv(outputfile,data_w)

        # print('采样各样本特征权重如下：')
        # print(self.W)
        # print('平均特征权重如下：')
        self.W = self.get_norm(self.W.copy())
        # print(self.W)

        return self.W


    def get_norm(self, W):
        # W = [0 if w < 0 else w for w in W]
        w_squared = np.square(W)
        w_norm_inf = np.linalg.norm(w_squared, ord=np.inf)
        normalized_w = w_squared / w_norm_inf
        return normalized_w

        # 返回最终选取的特征

    def get_final(self, cooling_rate):
        print('******************进入特征选择环节*******************')
        print('**************************************************')
        x = self.X
        y = self.Y

        a = NSGA2_simba(30, 300, len(self.W), self.W.copy(), x, y)
        final = a.get_subset()

        return final[0]


if __name__ == '__main__':
    a=0
    # # Read data from CSV file
    # df = pd.read_csv('../data/xor_dataset.data')
    #
    # # Extract features (X) and labels (Y) from the DataFrame
    # X = df.iloc[:, :-1]  # Extract all columns except the last one as features
    # Y = df.iloc[:, -1]  # Extract the last column as labels
    #
    #
    # al=Simba_plus(X,Y,0.63,1,0)
    # al.reliefF('../out/xor/simba_sample_0.2.csv')


    # print(al.process_discrete_features())

