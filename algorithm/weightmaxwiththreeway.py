
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
from scipy.stats import weibull_min
from sklearn.preprocessing import LabelEncoder

from algorithm import EVM
from algorithm.Nsga2 import NSGA2
from algorithm.Nsga2_simba import NSGA2_simba
from config import config
from utiles import utiles

'''
适用于多分类问题
'''
class weightmaxwiththreeway:
    def __init__(self, X,Y, T, lamada, k):
        self.X = X
        self.Y=Y
        self.T=T
        self.lamada=lamada
        self.X=self.process_discrete_features(self.X)
        self.label_counts=Y.value_counts()

        self.__k = k

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
        self.weight=[1]*len(X)
        self.get_all_neight()


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

        argmin_near = np.argsort(weighted_diff[same_class_indices])[:k]

        nearest_same_class_sample_index = same_class_indices[argmin_near]

        argmin_diff = np.argsort(weighted_diff[different_class_indices])[:k]
        nearest_different_class_sample_index = different_class_indices[argmin_diff]

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

    def getabsDiswithWeight(self,i,h):
        return (list(abs((self.X.iloc[i] - self.X.iloc[h]))))
    def getDiswithWeight(self,i,h):
        return (np.linalg.norm(self.W * (self.X.iloc[i] - self.X.iloc[h])))

    def getDiswithWeight_w(self,i,h,w):
        return (np.linalg.norm(w * (self.X.iloc[i] - self.X.iloc[h])))

    def get_weight_plus(self, feature, index, NearHit, NearMiss):

        nh = [(pow(self.getFeatureDis(index,h,feature),2)
              /(self.getDiswithWeight(index,h)))
              if not (self.getDiswithWeight(index,h)==0) else 0
              for h in NearHit]

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

    def get_all_neight(self):
        for i in range(len(self.X)):
            self.nearhit[i],self.nearmiss[i]=self.find_nearest_neighbors_plus(i,self.init_w,self.__k)

    #获取初始的margin矩阵
    def get_margin(self):
        for i in range(len(self.X)):
            self.find_nearest_neighbors(i)
        self.margin_init=self.margin.copy()

        return sum(self.margin_init)

    # 获取初始的nearmisdis矩阵
    def get_nearmisdis_with_w(self):
        nearmisdis=[self.getDiswithWeight(i, self.nearmiss[i][0])
                    -self.getDiswithWeight(i, self.nearhit[i][0])
                    for i in range(len(self.X))]

        return nearmisdis

    # 获取初始的margin矩阵
    def get_margin_with_w(self, w1):
        margin=0
        for i in range(len(self.X)):

            nm = [(self.getDiswithWeight_w(i, h,w1))
                  for h in self.nearmiss[i]]

            nh=[(self.getDiswithWeight_w(i, h,w1))
                  for h in self.nearhit[i]]

            k_nm = (self.weight[i]) / (len(nm))
            k_nh = (self.weight[i]) / (len(nh))
            sum_nm = sum(nm) * k_nm
            sum_nh = sum(nh) * k_nh
            margin+=(sum_nm-sum_nh)*(1/2)

        return margin

    def get_margin_matix(self):
        nm_margin=[]
        nh_margin=[]
        for i in range(len(self.X)):
            length=1/len(self.nearmiss[i])
            nm = [(self.getabsDiswithWeight(i, h))
                  for h in self.nearmiss[i]]
            nh=[(self.getabsDiswithWeight(i, h))
                  for h in self.nearhit[i]]
            nm_vec=np.array(nm)*length*self.weight[i]
            nm_margin.append(nm_vec)
            nh_vec=np.array(nh)*length*self.weight[i]
            nh_margin.append(nh_vec)
        return np.array(nm_margin),np.array(nh_margin)

    def get_margin_with_w1(self, w1):
        nm,nh=self.nm, self.nh

        margin=[np.sum(np.linalg.norm(nm[i]*w1, axis=1))
                - np.sum(np.linalg.norm(nh[i]*w1, axis=1))
                for i in range(len(self.X))]
        # nm_row_norms = np.linalg.norm(nm*w1, axis=1)
        # # 计算范数之和
        # nm_sum_of_norms = np.sum(nm_row_norms)
        # nh_row_norms = np.linalg.norm(nh * w1, axis=1)
        # # 计算范数之和
        # nh_sum_of_norms = np.sum(nh_row_norms)
        # return (nm_sum_of_norms-nh_sum_of_norms)*(1/2)
        return np.sum(margin)*(1/2)

    def get_normalized_weights(self,w):
        total_weight = sum(w)
        normalized_weights = [weight / total_weight for weight in w]
        return normalized_weights

    def choose_region(self,normalized_means):

        r = random.random()

        if r < normalized_means[0]:
            return 'bound'
        elif r < normalized_means[0] + normalized_means[1]:
            return 'mid'
        else:
            return 'far'

    def choose_sample_index(self,region_indices, region_weights):
        return random.choices(region_indices, weights=region_weights,k=1)[0]

    def calculate_sample_means(self,w):
        sorted_indices = np.argsort(w)  # Sort in descending order
        bound_end = int(len(w) * 0.25)
        mid_start = bound_end
        mid_end = int(len(w) * 0.75)
        # bound_indices = sorted_indices[:bound_end]
        # mid_indices = sorted_indices[mid_start:mid_end]
        # far_indices = sorted_indices[mid_end:]
        bound_indices = sorted_indices[mid_end:]
        mid_indices = sorted_indices[mid_start:mid_end]
        far_indices = sorted_indices[:bound_end]

        # print(w[bound_indices])
        # print(w[mid_indices])
        # print(w[far_indices])


        mean_bound = np.mean(w[bound_indices])
        mean_mid = np.mean(w[mid_indices])
        mean_far = np.mean(w[far_indices])
        # print(mean_far)
        # print(mean_mid)
        # print(mean_bound)

        return mean_bound, mean_mid, mean_far, bound_indices, mid_indices, far_indices

    def chooseidex(self,w, bound_indices, mid_indices, far_indices,normalized_means):
        chosen_region = self.choose_region(normalized_means)
        # print('选择的是：',chosen_region)
        if chosen_region == 'bound':
            chosen_index = self.choose_sample_index(bound_indices, w[bound_indices])
        elif chosen_region == 'mid':
            chosen_index = self.choose_sample_index(mid_indices, w[mid_indices])
        else:
            chosen_index = self.choose_sample_index(far_indices, w[far_indices])
        return chosen_index

    # 过滤式特征选择
    def fit(self):
        equal_val=1e-5
        m, n = np.shape(self.X)  # m为行数，n为列数

        data_w = []
        data_w.append([1] * n)
        margin=self.get_nearmisdis_with_w()
        shape, loc, scale = weibull_min.fit(margin, floc=0)
        # self.weight = EVM.get_prob(self.get_nearmisdis_with_w(), self.Y)
        cdf = weibull_min.cdf(margin, shape, loc, scale)

        self.weight = [1 - cdf[i] for i in range(len(cdf))]
        mean_bound, mean_mid, mean_far, bound_indices, mid_indices, far_indices = self.calculate_sample_means(np.array(self.weight))
        # print(mean_bound)
        # print(mean_mid)
        # print(mean_far)
        # print(bound_indices)
        # print(mid_indices)
        # print(far_indices)

        normalized_means = self.get_normalized_weights([mean_bound, mean_mid, mean_far])

        # print(normalized_means)
        self.nm, self.nh=self.get_margin_matix()
        # old_margin = self.get_margin_with_w(self.W)

        # self.nm, self.nh = self.get_margin_matix()
        # print(self.get_margin_with_w(self.W))
        old_margin = self.get_margin_with_w1(self.W)
        # print('old margin:',old_margin)

        self.marginadd = []
        random.seed(42)
        t=0
        best_equal=float('inf')
        # for t in range(self.T):
        while(True):
            t=t+1
            print('t:', t)
            # selected_sample = random.choices(range(0, m), weights=self.weight, k=1)[0]
            selected_sample =self.chooseidex(np.array(self.weight), bound_indices, mid_indices, far_indices, normalized_means)

            # print('样本权重：',self.weight[selected_sample])

            e=[0 for f in self.X.columns]

            for f in self.X.columns:
                f_index = self.X.columns.get_loc(f)

                e[f_index] += ((self.get_weight_plus
                                (f_index, selected_sample,
                                 self.nearhit[selected_sample],
                                 self.nearmiss[selected_sample])))

            w_range=[e[f]*self.W[f]*(1/2)*self.lamada for f in range(len(self.X.columns))]

            w_copy=self.W.copy()

            W_new=[w_copy[i]+w_range[i] for i in range(len(w_copy))]

            new_margin = self.get_margin_with_w1(W_new)
            # new_margin = self.get_margin_with_w1(W_new)
            # print(new_margin)
            if np.isnan(new_margin):
                break

            self.W = W_new
            beequal=abs(new_margin-old_margin)
            if(beequal<best_equal):
                best_equal=beequal
            print(best_equal)
            # print(old_margin)
            if beequal<equal_val :
                print('算法收敛')
                break
            else:
                old_margin=new_margin
        self.W = self.get_norm(self.W.copy())

        # print('w is   :    ',self.W)

        return self.W,t



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

    def feature_ranking(self):
        idx = np.argsort(self.W, 0)
        return idx[::-1]

    def get_subset(self):
        w=self.W.copy()
        # 创建长度为 len_vector 的零向量
        subset = np.zeros(len(w))
        idx=self.feature_ranking()
        subset[idx[0]]=1
        max_magin = self.get_margin_with_w1(subset)
        max_subset = subset
        while True:
            flag=False
            subsetcopy = subset.copy()
            for index in idx:
                if subset[index] == 1:
                    continue
                subsetcopy[index]=1
                margin=self.get_margin_with_w1(subsetcopy)
                if margin>max_magin:
                    flag=True
                    max_magin=margin
                    max_subset=subsetcopy.copy()
                subsetcopy[index]=0
            if(flag):
                subset=max_subset.copy()
            else:
                break
        return subset


    def get_final(self,cooling_rate):
        print('******************进入特征选择环节*******************')
        print('**************************************************')
        x = self.X
        y = self.Y

        a = NSGA2_simba(10, 100, len(self.W), self.W.copy(), x, y)
        final = a.get_subset()

        return final[0]



if __name__ == '__main__':
    a=0
    # x,y=utiles.get_scaler_data('../data/final/xor_dataset.data')
    # a=evSimba_plus1(x,y,0.01,1,1)
    # a.reliefF(30,5,'../out/exmp2/xor_w.csv')
    # # a.get_final(0.2)

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

