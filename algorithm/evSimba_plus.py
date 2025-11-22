
import math
import sys
from collections import defaultdict
import time

import pandas as pd
import numpy as np
import numpy.linalg as la
import random
import csv

from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

from algorithm import EVM
from algorithm.Nsga2 import NSGA2
from config import config
from utiles import utiles

'''
适用于多分类问题
'''
class evSimba_plus:
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
        self.init_w=[1]*((X.shape[1]))

        self.nearhit={}
        self.nearhitdis={}
        self.nearmissdis=[]
        self.nearmiss={}
        self.margin={}
        self.e={}
        self.nearhit_feature_dis = defaultdict(dict)
        self.nearmiss_feature_dis= defaultdict(dict)

        self.initial_temperature = config.initial_temperature
        # self.cooling_rate = config.cooling_rate
        #权重值
        self.weight=[1]*len(X)
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

        same_class_indices = np.where(Y == Y.iloc[i])[0]

        same_class_indices = same_class_indices[same_class_indices != i]

        different_class_indices = np.where(Y != Y.iloc[i])[0]

        argmin_near = np.argmin(weighted_diff[same_class_indices])
        nearest_same_class_sample_index = same_class_indices[argmin_near]
        argmin_diff = np.argmin(weighted_diff[different_class_indices])
        nearest_different_class_sample_index = different_class_indices[argmin_diff]

        return nearest_same_class_sample_index, nearest_different_class_sample_index



    def find_nearest_neighbors_plus(self, i,w,k):

        X = self.X
        Y = self.Y
        weights = w

        weighted_diff = np.linalg.norm(weights * (X - X.iloc[i]), axis=1)

        same_class_indices = np.where(Y == Y.iloc[i])[0]

        same_class_indices = same_class_indices[same_class_indices != i]

        different_class_indices = np.where(Y != Y.iloc[i])[0]

        # y = np.unique((Y.values[different_class_indices]))


        argmin_near = np.argsort(weighted_diff[same_class_indices])[:k]

        nearest_same_class_sample_index = same_class_indices[argmin_near]


        # yi_index = np.where(not(Y == Y.iloc[i]))[0]
        argmin_diff = np.argsort(weighted_diff[different_class_indices])[:k]
        nearest_different_class_sample_index = different_class_indices[argmin_diff]

        # for yi in y:
        #     yi_index=np.where(Y == yi)[0]
        #     argmin_diff = np.argsort(weighted_diff[yi_index])[:self.__k]
        #     nearest_different_class_sample_index[yi] = yi_index[argmin_diff]



        return nearest_same_class_sample_index, nearest_different_class_sample_index


    def find_nearest_neighbors_with_W(self, i,W):
        # 记录开始时间
        # start_time = time.time()
        X=self.X
        Y=self.Y
        # self.W=W
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


        self.nearhit[i]=nearest_same_class_sample_index

        self.nearmiss[i]=nearest_different_class_sample_index

        self.nearhitdis[i]=round(nearest_same_class_sample_distance,2)
        self.nearmissdis[i]=round(nearest_different_class_sample_distance,2)

        self.margin[i]=(self.nearmissdis[i]-self.nearhitdis[i])*(1/2)*self.weight[i]

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


    def getProb(self, i):
        Y = self.Y

        same_class_indices = np.where(Y == Y.iloc[i])[0]

        return len(same_class_indices) / len(Y)

    def getFeatureDis(self,i,h,feature):
        return abs(self.X.iloc[i, feature] - self.X.iloc[h, feature])


    def getDiswithWeight(self,i,h):
        return (np.linalg.norm(self.W * (self.X.iloc[i] - self.X.iloc[h])))

    def getDiswithWeight_w(self,i,h,w):
        return (np.linalg.norm(w * (self.X.iloc[i] - self.X.iloc[h])))

    def get_weight_plus(self, feature, index, NearHit, NearMiss):

        nh = [(pow(self.getFeatureDis(index,h,feature),2)
              /(self.getDiswithWeight(index,h))) if not (self.getDiswithWeight(index,h)==0) else 0  for h in NearHit]

        k_nh = (self.weight[index])/ (len(nh))
        sum_nh = k_nh * sum(nh)

        # pi = self.getProb(index)

        nm = [(pow(self.getFeatureDis(index,h,feature),2)
              /(self.getDiswithWeight(index,h)))
              if not (self.getDiswithWeight(index,h)==0) else 0
              for h in NearMiss]

        k_nm = (self.weight[index]) / (len(nm))
        sum_nm = sum(nm) * k_nm

        return sum_nm - sum_nh


    #获取初始的margin矩阵
    def get_margin(self):
        for i in range(len(self.X)):
            self.find_nearest_neighbors(i)
        self.margin_init=self.margin.copy()

        return sum(self.margin_init)

    # 获取初始的nearmisdis矩阵
    def get_nearmisdis_with_w(self):
        nearmisdis=[]
        for i in range(len(self.X)):

            pi = self.getProb(i)
            NearHit, NearMiss=self.find_nearest_neighbors_plus(i,self.init_w,1)

            sum_nm = self.getDiswithWeight(i, NearMiss)
            nearmisdis.append(sum_nm)

        return nearmisdis

    # 获取初始的margin矩阵
    def get_margin_with_w(self, w1,w2):

        margin=0
        for i in range(len(self.X)):
            pi = self.getProb(i)
            NearHit, NearMiss = self.find_nearest_neighbors_plus(i, w2,self.__k)

            # nm = [(self.getDiswithWeight_w(i, h,w1)) * (self.getProb(h) / (1 - pi))
            #       for h in NearMiss]
            nm = [(self.getDiswithWeight_w(i, h,w1))
                  for h in NearMiss]

            nh=[(self.getDiswithWeight_w(i, h,w1))
                  for h in NearHit]

            k_nm = (self.weight[i]) / (len(nm))
            k_nh = (self.weight[i]) / (len(nh))
            sum_nm = sum(nm) * k_nm
            sum_nh = sum(nh) * k_nh

            margin+=(sum_nm-sum_nh)


        return margin


    # 过滤式特征选择
    def reliefF(self, T,t_Close,outputfile='../../out/exmp2/xor_w.csv', sample_way_isSampleWeight=True):

        m, n = np.shape(self.X)  # m为行数，n为列数

        data_w = []
        data_w.append([1] * n)


        self.weight = EVM.get_prob(self.get_nearmisdis_with_w(), self.Y)

        old_margin=float('-inf')

        flag=False
        t_while=0

        max_margin=float('-inf')
        max_w=self.W.copy()


        self.marginadd = []

        isClose=False
        t_isClose=0
        close=0.01
        # self.marginadd.append(old_margin)

        while(True):

            print('t:', t_while)

            sample_index = random.sample(range(0, m), self.__sample_num)

            e=[0 for f in self.X.columns]


            for i in sample_index:  # 采样次数

                NearHit, NearMiss = self.find_nearest_neighbors_plus(i,self.W,self.__k)

                for f in self.X.columns:

                    f_index = self.X.columns.get_loc(f)

                    e[f_index]+=(round(self.get_weight_plus(f_index,i,NearHit,NearMiss), 2))


            w_range=[e[f]*self.W[f]*(1/self.__sample_num) for f in range(len(self.X.columns))]

            w_copy=self.W.copy()

            W_new=[w_copy[i]+w_range[i] for i in range(len(w_copy))]

            W_new_norm = self.get_norm(W_new.copy())

            new_margin = self.get_margin_with_w(W_new_norm,W_new_norm)

            if np.isnan(new_margin):
                break

            if new_margin>max_margin :
                max_margin=new_margin
                max_w=W_new

            # print(W_new)
            # print(self.W)
            print('old margin:', old_margin)
            print('new margin:', new_margin)

            if ((t_while > T) or (t_isClose >= t_Close and isClose)):

                print('终止条件：')
                if (t_isClose >= 10):
                    print(f'margin值在趋于收敛后迭代{t_Close}次后未发现有提升情况')
                else:
                    print('迭代次数超过：', T)
                    if (not flag):
                        print('未能找到很好的w，能够使得其margin大于最初的w')
                        self.W = max_w
                    else:
                        print('找到当前最大的margin值，最大margin值在循环超过次数后还是没有变化')

                data_w.append(self.get_norm(self.W.copy()))
                break


            elif((old_margin-new_margin)<(-close)):
                print('old<new')

                if isClose:
                    t_isClose=0
                    isClose=False

                flag = True
                old_margin = new_margin

                self.marginadd.append(old_margin)
                self.W = W_new

                data_w.append(W_new_norm)


            elif (old_margin - new_margin>close):
                print('old>new')

                if isClose:
                    t_isClose = 0
                    isClose = False
                    self.marginadd.append(old_margin)

            else:

                print('收敛状态')

                if not isClose:
                    isClose = True
                    t_isClose = 0

                else:
                    t_isClose+=1

                if old_margin<new_margin:
                    old_margin=new_margin
                    self.marginadd.append(new_margin)
                    self.W = W_new
                    data_w.append(W_new_norm)
            t_while+=1





        self.W = self.get_norm(self.W.copy())

        print('w is   :    ',self.W)

        utiles.array_to_csv(outputfile, data_w)

        return self.W



    def get_norm(self, W):
        # W = [0 if w < 0 else w for w in W]
        w_squared = np.square(W)

        w_norm_inf = np.linalg.norm(w_squared, ord=np.inf)

        normalized_w = w_squared / w_norm_inf

        return normalized_w

        # 返回最终选取的特征


    def get_margin_with_subset(self,w):

        X = self.X
        margin=0
        for i in range(len(X)):

            nn_dis = np.linalg.norm(w * (X.iloc[i] - X.iloc[self.NH[i]]))*self.weight[i]

            nm_dis = np.linalg.norm(w * (X.iloc[i] - X.iloc[self.NM[i]]))*self.weight[i]
            # print(X.iloc[i])
            # print(i)
            # print(self.NM[i])
            # print(X.iloc[self.NM[i]])
            # print(w)
            # print(nm_dis)
            #
            # print(margin)
            margin += (round(((nm_dis - nn_dis) * (1 / 2)), 2))


        return margin

    def get_final(self,cooling_rate):
        print('******************进入特征选择环节*******************')
        print('**************************************************')
        x = self.X
        y = self.Y

        a = NSGA2(30, 300, len(self.W), self.W.copy(), x, y)
        final = a.get_subset()

        return final[0]



if __name__ == '__main__':
    a=0
    x,y=utiles.get_scaler_data('../data/final/xor_dataset.data')
    a=evSimba_plus(x,y,0.01,1,1)
    a.reliefF(30,5,'../out/exmp2/xor_w.csv')
    # a.get_final(0.2)

    # a.weight=EVM.get_prob(a.get_nearmisdis_with_w(), a.Y)
    # # a.reliefF()
    #
    # import pandas as pd
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # # 假设 x 包含数据，y 包含类别，weight 包含权重
    # # x 是一个 DataFrame，包含两列数据
    # # y 是一个 Series，包含每个数据点的类别
    # # weight 是一个 NumPy 数组，包含每个数据点的权重
    #
    # # 创建一个图形
    # plt.figure(figsize=(8, 6))
    #
    # # 创建颜色映射
    # colors = ['b','r','y']
    # shapes = ['o', 's', '^', 'v']  # 可以根据需要添加更多形状
    #
    # # 获取唯一的类别
    # unique_labels = y.unique()
    #
    # # 绘制数据分布
    # for label in unique_labels:
    #     indices = y[y == label].index
    #     # size = np.array(a.weight)[indices] * 100  # 根据权重调整点的大小
    #     size = 50  # 根据权重调整点的大小
    #     alpha = np.array(a.weight)[indices]  # 根据权重设置透明度
    #     color = colors
    #     shape = shapes[label % len(shapes)]  # 根据类别选择形状
    #     plt.scatter(x.loc[indices, 0], x.loc[indices, 1], c=color[label], marker=shape, alpha=alpha, s=size, label=f'Class {label}')
    #
    # # 添加图例
    # plt.legend()
    #
    # # 添加标签
    # plt.xlabel('Feature 0')
    # plt.ylabel('Feature 1')
    #
    # # 显示图像
    # # plt.show()
    #
    # # 显示图像
    # plt.savefig('../img/final_exmp/sampleweight.jpg')
    # plt.show()
    # plt.close()
    # # # Read data from CSV file
    #
    #
    # #
    # # # Extract features (X) and labels (Y) from the DataFrame
    # X,Y=utiles.get_scaler_data('../data/rbf_data.csv')
    # #
    # #
    # al=Simba_sample_weight_sigmoid(X,Y,0.2,1,0)
    # al.reliefF('../out/xor/simba_sample_sigmoid_0.2.csv')


    # print(al.process_discrete_features())

