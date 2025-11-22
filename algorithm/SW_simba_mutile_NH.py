
import math
import sys
from collections import defaultdict
import time

import pandas as pd
import numpy as np
import numpy.linalg as la
import random
import csv

from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder


from utiles import utiles

'''
适用于多分类问题
'''
class SW_simba_plus:
    def __init__(self, X,Y, sample_rate, t, k):
        self.X = X
        label_encoder = LabelEncoder()
        # self.Y = label_encoder.fit_transform(Y)
        self.Y=Y
        self.X=self.process_discrete_features(self.X)

        self.__sample_num = int(round(len(X) * sample_rate))
        self.__t = t

        self.label_counts=Y.value_counts()

        self.min_label_count=self.label_counts.min()
        self.__k = min(k,self.min_label_count)

        self.W=[1]*((X.shape[1]))

        self.nearhit={}
        self.nearhitdis={}
        self.nearmissdis={}
        self.nearmiss={}
        self.margin={}
        self.e={}
        self.nearhit_feature_dis = defaultdict(dict)
        self.nearmiss_feature_dis= defaultdict(dict)

        #权重值
        self.weight=[]
        self.pk_i={}

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

        same_class_indices = np.where(Y == Y[i])[0]

        same_class_indices = same_class_indices[
            same_class_indices != i]

        different_class_indices = np.where(Y != Y[i])[0]

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
        # X=self.X
        # Y=self.Y
        # weights=self.W
        # # print(Y)
        # # 初始化最近同类和异类样本的索引和距离
        # nearest_same_class_sample_index = None
        # nearest_same_class_sample_distance = float('inf')
        # nearest_different_class_sample_index = None
        # nearest_different_class_sample_distance = float('inf')
        #
        # for j in range(len(X)):
        #     if i == j:
        #         continue
        #     weighted_diff = np.linalg.norm(weights * (X.iloc[i] - X.iloc[j]))
        #     # print('weighted_diff ', weighted_diff)
        #     if Y[j] == Y[i] and weighted_diff < nearest_same_class_sample_distance:
        #         # print(weighted_diff)
        #         # print(j)
        #         nearest_same_class_sample_index = j
        #         nearest_same_class_sample_distance = weighted_diff
        #         # print('same ',nearest_same_class_sample_distance,weighted_diff,j)
        #
        #     elif Y[j] != Y[i] and weighted_diff < nearest_different_class_sample_distance:
        #         nearest_different_class_sample_index = j
        #         nearest_different_class_sample_distance = weighted_diff
        #         # print('different ',nearest_different_class_sample_distance)
        #
        #
        #
        # self.nearhit[i]=nearest_same_class_sample_index
        # self.nearmiss[i]=nearest_different_class_sample_index
        #
        # self.nearhitdis[i]=round(nearest_same_class_sample_distance,2)
        # self.nearmissdis[i]=round(nearest_different_class_sample_distance,2)
        # # print(nearest_same_class_sample_distance)
        # # print(nearest_different_class_sample_distance)
        # # sys.exit()
        #
        # self.margin[i]=(self.nearmissdis[i]-self.nearhitdis[i])*(1/2)
        #
        # return nearest_same_class_sample_index, nearest_different_class_sample_index

    def find_nearest_neighbors_plus(self, i):

        X = self.X
        Y = self.Y
        weights = self.W

        weighted_diff = np.linalg.norm(weights * (X - X.iloc[i]), axis=1)

        same_class_indices = np.where(Y == Y.iloc[i])[0]

        same_class_indices = same_class_indices[same_class_indices != i]

        different_class_indices = np.where(Y != Y.iloc[i])[0]

        y = np.unique((Y.values[different_class_indices]))


        argmin_near = np.argsort(weighted_diff[same_class_indices])[:self.__k]

        nearest_same_class_sample_index = same_class_indices[argmin_near]

        nearest_same_class_sample_distance = weighted_diff[same_class_indices[argmin_near]]

        sum_other=sum(self.label_counts)-len(same_class_indices)

        pi_k={ k:(round((len(np.where(Y == k)[0])/sum_other)*(1/self.__k),2)) for k in y }
        self.pk_i[i]=pi_k

        nearest_different_class_sample_distance={}
        nearest_different_class_sample_index={}

        for yi in y:
            yi_index=np.where(Y == yi)[0]
            argmin_diff = np.argsort(weighted_diff[yi_index])[:self.__k]

            nearest_different_class_sample_index[yi] = yi_index[argmin_diff]
            nearest_different_class_sample_distance[yi] = weighted_diff[yi_index[argmin_diff]]

        self.nearhit[i] = nearest_same_class_sample_index

        self.nearmiss[i] = nearest_different_class_sample_index

        self.nearhitdis[i] = nearest_same_class_sample_distance

        self.nearmissdis[i] = nearest_different_class_sample_distance

        fronter_nearmisdis={k:v*pi_k[k] for k,v in nearest_different_class_sample_distance.items()}
        # print(nearest_different_class_sample_distance)
        # print(pi_k)
        # print(fronter_nearmisdis)
        # sys.exit()

        pronter=sum([sum(value) for value in fronter_nearmisdis.values()])


        self.margin[i] = (pronter - sum(self.nearhitdis[i])*(1/self.__k)) * (1 / 2)
        # print(self.nearhit[i])
        # print(self.nearmiss[i])
        # print(self.nearhitdis[i])
        # print(self.nearmissdis[i])
        # sys.exit()

        return nearest_same_class_sample_index, nearest_different_class_sample_index


    def find_nearest_neighbors_with_W(self, i,W):
        X = self.X
        Y = self.Y
        weights = W

        weighted_diff = np.linalg.norm(weights * (X - X.iloc[i]), axis=1)

        same_class_indices = np.where(Y == Y.iloc[i])[0]

        same_class_indices = same_class_indices[same_class_indices != i]

        different_class_indices = np.where(Y != Y.iloc[i])[0]

        y = np.unique((Y.values[different_class_indices]))

        argmin_near = np.argsort(weighted_diff[same_class_indices])[:self.__k]

        nearest_same_class_sample_index = same_class_indices[argmin_near]

        nearest_same_class_sample_distance = weighted_diff[same_class_indices[argmin_near]]

        sum_other = sum(self.label_counts) - len(same_class_indices)

        pi_k = {k: (round((len(np.where(Y == k)[0]) / sum_other) * (1 / self.__k), 2)) for k in y}
        self.pk_i[i] = pi_k

        nearest_different_class_sample_distance = {}
        nearest_different_class_sample_index = {}

        for yi in y:
            yi_index = np.where(Y == yi)[0]
            argmin_diff = np.argsort(weighted_diff[yi_index])[:self.__k]

            nearest_different_class_sample_index[yi] = yi_index[argmin_diff]
            nearest_different_class_sample_distance[yi] = weighted_diff[yi_index[argmin_diff]]

        self.nearhit[i] = nearest_same_class_sample_index

        self.nearmiss[i] = nearest_different_class_sample_index

        self.nearhitdis[i] = nearest_same_class_sample_distance

        self.nearmissdis[i] = nearest_different_class_sample_distance

        fronter_nearmisdis = {k: v * pi_k[k] for k, v in nearest_different_class_sample_distance.items()}

        pronter = sum([sum(value) for value in fronter_nearmisdis.values()])

        self.margin[i] = (pronter - sum(self.nearhitdis[i]) * (1 / self.__k)) * (1 / 2)


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

    def get_weight_plus(self, feature, index):

        data = self.X
        row = data.iloc[index]

        right_w_up = pow((data.iloc[self.nearhit[index]][feature]-row[feature]),2)*(1/self.__k)

        right_w_down=self.nearhitdis[index]
        right_w=[ ((right_w_up.values[i]/right_w_down[i]) if right_w_down[i]!=0 else 0 ) for i in range(len(right_w_down))]

        #后半部分代码
        right_w_value=sum(right_w)

        # print(right_w_value)
        # print(self.nearmiss[index])

        #计算公式前者代码
        wrong_w=0
        for k, v in self.nearmiss[index].items():
            # print(k)
            # print(v)
            nearmiss = data.iloc[v]
            # print(nearmiss)
            # print(self.pk_i)
            wrong_w_up = pow((nearmiss[feature] - row[feature]), 2) * (1 / self.__k)*(self.pk_i[index][k])
            # print(nearmiss)
            # print((nearmiss[feature]))
            # print(row[feature])
            # print((nearmiss[feature] - row[feature]))
            # print(pow((nearmiss[feature] - row[feature]), 2))
            # print(wrong_w_up)
            wrong_w_down =self.nearmissdis[index][k]

            wrong_w_k = [((wrong_w_up.values[i] / wrong_w_down[i]) if wrong_w_down[i] != 0 else 0) for i in
                       range(len(wrong_w_down))]

            wrong_w+= sum(wrong_w_k)

            # print(wrong_w_down)
            # print(wrong_w_k)
            # print(wrong_w)




        # print(self.nearmiss[index])
        # print(self.nearmiss[index].values())



        # wrong_w = pow(round(abs(row[feature] - nearmiss[feature]), 2), 2)



        # fronter = 0 if (self.nearmissdis[index] == 0) else (wrong_w / self.nearmissdis[index])

        # laster = 0 if (self.nearhitdis[index] == 0) else (right_w / self.nearhitdis[index])

        self.e[index]=round(wrong_w - right_w_value,2)*(1/2)


        self.nearhit_feature_dis[index][feature]=right_w_value

        self.nearmiss_feature_dis[index][feature]=wrong_w
        # print(self.e[index])
        # sys.exit()
        return right_w_value,wrong_w

    #获取新的margin特征
    def calculate_new_features(self):

        X=self.X
        Y=self.Y
        new_features = []  # 存储新的特征
        for i, xi in X.iterrows():
            class_label = Y.loc[i]  # 当前数据行的类标签
            same_class_indices = Y[Y == class_label].index.drop(i)  # 去除当前下标
            other_class_indices = Y[Y != class_label].index  # 异类的下标

            new_feature_values = []  # 存储新的特征值
            for feature_name, xij in xi.iteritems():

                # 计算当前特征值 xij 减去每个异类在该特征下的特征值的绝对值，然后求和
                sum_other_margin = np.abs(xij - X.loc[other_class_indices, feature_name]).sum()

                # 计算当前特征值 xij 减去每个同类在该特征下的特征值的绝对值，然后求和
                sum_same_margin = np.abs((xij - X.loc[same_class_indices, feature_name])).sum()

                # 计算新的特征值
                new_feature_value = sum_other_margin - sum_same_margin
                new_feature_values.append(new_feature_value)

            new_features.append(new_feature_values)

        # 构建新的DataFrame，列名以原特征名加上'_new'后缀命名
        new_df = pd.DataFrame(new_features, columns=X.columns)
        return new_df

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def tanh_scaling(self,x, scale_factor=10):
        return (np.tanh(x * scale_factor) + 1) / 2
    #计算Sample_weight
    def cal_Sample_weight(self):
        margin=self.calculate_new_features()
        distances = cdist(margin, margin, metric='euclidean')
        mean=np.mean(distances,axis=1)
        reciprocal_array = [1 / x for x in mean]

        sum_dis=sum(reciprocal_array)
        weight=[x/sum_dis for x in reciprocal_array]

        # 使用 NumPy 的 sigmoid 函数将数组元素映射成 sigmoid 函数的值
        # probabilities = self.tanh_scaling(np.array(weight),100)
        min_va=np.min(weight)
        max_va=np.max(weight)
        min_max_normalized_array = (weight - min_va) / (max_va - min_va)
        self.weight=min_max_normalized_array
        return self.weight




    # 过滤式特征选择
    def reliefF(self,outputfile='../out/xor/xor_w.csv',sample_way_isSampleWeight=True):

        # print sample
        m, n = np.shape(self.X)  # m为行数，n为列数
        data_w = []

        #获取权重值：
        self.cal_Sample_weight()
        print('获取完样本的权重值')
        print('样本的权重值:',self.weight)
        #判断是否是按照样本权重采样
        if sample_way_isSampleWeight:
            # 如果按照样本权重采样，则通过赋予不同位置上的权重。然后按照权重重复采样
            sample_index = utiles.get_sample_with_weigth(self.weight,m,self.__sample_num)
        else:
            #否则按照随机取样方法进行
            sample_index=random.sample(range(0, m), self.__sample_num)



        for i in sample_index:  # 采样次数

            try:
                NearHit, NearMiss = self.find_nearest_neighbors_plus(i)
            except:
                continue

            # print('第 %s 次采样，样本index为 %s，其NearHit k近邻行索引为 %s ，NearMiss k近邻行索引为 %s' % (num, i, NearHit, NearMiss))
            for f in self.X.columns:
                # print('***:', f, i, NearHit, NearMiss)

                self.get_weight_plus(f, i)

                f_index = self.X.columns.get_loc(f)
                w_range = round(self.e[i] * (self.W[f_index]), 2)
                self.W[f_index] += w_range

                # print('特征 %s 的权重变化为 %s.' % (f, w_range))
            # score.append(one_score)

            data_w.append(self.get_norm(self.W.copy()))

        utiles.array_to_csv(outputfile,data_w)

        print('采样各样本特征权重如下：')
        print(self.W)
        print('平均特征权重如下：')
        self.W = self.get_norm(self.W.copy())
        print(self.W)

        return self.W


    def get_norm(self, W):
        # W = [0 if w < 0 else w for w in W]
        w_squared = np.square(W)
        w_norm_inf = np.linalg.norm(w_squared, ord=np.inf)
        normalized_w = w_squared / w_norm_inf
        return normalized_w

        # 返回最终选取的特征

    def get_final(self):

        f_w = pd.DataFrame(self.W, columns=['weight'])
        print(f_w)
        m, n = np.shape(self.__data)
        final_feature_t = f_w[f_w['weight'] > self.__t]
        # print('*' * 100)
        # print(final_feature_t)
        # final_feature_k = f_w.sort_values('weight').tail(int((n - 1) * 0.5))
        # print(final_feature_k)
        # final_feature_k = f_w.sort_values('weight').head(self.__k)
        # print final_feature_k
        return final_feature_t


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

