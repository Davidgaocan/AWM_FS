
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
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from algorithm import EVM
from config import config
from utiles import utiles

'''
适用于多分类问题
'''
class evrelief_plus1:
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

        self.n = X.shape[1]
        self.W=[1/self.n]*self.n

        self.pairDistance_drop = []
        self.pairDistance = []


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

        self.nnmodel = EVM.get_nnmodel(self.getAllPointPairDistance())


    # 数据处理（将离散型数据处理成连续型数据，比如字符到数值）
    def process_discrete_features(selif,x):
        new_dataframe = x.copy()
        label_encoder = LabelEncoder()
        for column in new_dataframe.columns:
            if new_dataframe[column].dtype == 'object':
                new_dataframe[column] = label_encoder.fit_transform(new_dataframe[column])

        return new_dataframe

    def find_nearest_neighbors_plus(self, i,w,k):

        X = self.X
        Y = self.Y
        weights = w
        weighted_diff = np.linalg.norm(weights * (X - X.iloc[i]), axis=1)
        same_class_indices = np.where(Y == Y.iloc[i])[0]
        same_class_indices = same_class_indices[same_class_indices != i]
        different_class_indices = np.where(Y != Y.iloc[i])[0]
        # y = np.unique((Y.values[different_class_indices]))
        argmin_near = np.argsort(weighted_diff[same_class_indices])

        nearest_same_class_sample_index = same_class_indices[argmin_near]

        # yi_index = np.where(not(Y == Y.iloc[i]))[0]
        argmin_diff = np.argsort(weighted_diff[different_class_indices])
        nearest_different_class_sample_index = different_class_indices[argmin_diff]


        return nearest_same_class_sample_index, nearest_different_class_sample_index

    def getOnePointPairDistance(self,i):

        X = self.X

        weighted_diff = (self.W*abs((X - X.iloc[i])))
        weighted_drop = (self.W*abs((X - X.iloc[i]))).drop(i)

        return np.array(weighted_drop.sum(axis=1)),np.array(weighted_diff.sum(axis=1))

    def getAllPointPairDistance(self):

        for i in range(len(self.X)):
            arr,arr2=self.getOnePointPairDistance(i)
            self.pairDistance_drop.append(arr)
            self.pairDistance.append(arr2)
        return self.pairDistance_drop

    def getNNModel(self):
        self.model = EVM.get_nnmodel(self.getAllPointPairDistance())

    def getProb(self, i):
        Y = self.Y

        same_class_indices = np.where(Y == Y.iloc[i])[0]

        return len(same_class_indices) / len(Y)

    def getFeatureDis(self,i,h,feature):
        return abs(self.X.iloc[i, feature] - self.X.iloc[h, feature])


    def getDiswithWeight(self,i,h):
        # return (np.linalg.norm(self.W * (self.X.iloc[i] - self.X.iloc[h])))
        nm=(abs(self.X.iloc[h]-self.X.iloc[i] )*self.W).sum(axis=1)
        # print(abs(self.X.iloc[h]-self.X.iloc[i]))
        # print(self.W)
        # print(nm)
        nm=np.mean(nm)

        # print(nm)
        return nm

    def getDiswithWeight_w(self,i,h,w):
        return (np.linalg.norm(w * (self.X.iloc[i] - self.X.iloc[h])))

    def getonePointDistanceWithW(self,i):
        X=self.X
        weighted_diff = self.W*abs((X - X.iloc[i]))
        return np.array(weighted_diff.sum(axis=1))

    def getProwithNNmodel(self,i):
        margin=self.pairDistance[i]
        return EVM.get_pro_with_nnmodel(self.model,i,margin)

    def get_one_z(self, index,w):
        x=self.X
        NearHit, NearMiss=self.find_nearest_neighbors_plus(index,w,self.__k)
        probs=np.array(self.getProwithNNmodel(index))

        probmis=sum(probs[NearMiss])
        probhit=sum(probs[NearHit])

        prob_out=probhit/(probmis+probhit) if (probmis+probhit)!=0 else 0
        probs=[(probs[i]/probmis) if i in NearMiss else (probs[i]/probhit) for i in range(len(x))]
        probs=np.array(probs)
        probs_NearMiss=probs[NearMiss]
        probs_NearHit=probs[NearHit]
        x_nm=(probs_NearMiss[:, np.newaxis]*abs(x.iloc[NearMiss]-x.iloc[index])).sum(axis=0)
        x_nh=(probs_NearHit[:, np.newaxis]*abs(x.iloc[NearHit]-x.iloc[index])).sum(axis=0)

        # print(x_nm)
        # print(probs[NearMiss])
        # print(probs[113])

        # print(x_nm)
        # print((x_nm-x_nh))
        n_margin=(x_nm-x_nh)*self.weight[index]*prob_out
        # print(n_margin)
        # print(n_margin)
        # sys.exit()
        return np.array(n_margin)


    def get_metix_Z(self,w):
        z=[0]*(self.n)
        # range(len(self.X))
        for i in range(len(self.X)):
            one=self.get_one_z(i,w)
            z=z+one
        z=z*(1/len(self.X))

        print('z is :', z)
        z=[0 if zi<0 else zi for zi in z]
        print(z)
        z_norm=(np.linalg.norm(z))

        z=[(zi/z_norm) for zi in z] if z_norm>0 else [0]*self.n

        return z

    def get_metix_Zwith_w(self,w):
        z=[0]*(self.n)
        # range(len(self.X))
        for i in range(len(self.X)):
            one=self.get_one_z(i,w)
            z=z+one
        z=z*(1/len(self.X))
        return z

    # 获取初始的nearmisdis矩阵
    def get_nearmisdis_with_w(self,t=1):
        nearmisdis=[]
        for i in range(len(self.X)):
            NearHit, NearMiss=self.find_nearest_neighbors_plus(i,self.W,1)

            sum_nm = self.getDiswithWeight(i, NearMiss)

            nearmisdis.append(sum_nm*t)


        return nearmisdis

    # 获取初始的margin矩阵
    def get_objectFunction(self, w,z):


        return np.dot(w, z.T)



    def compare_convergence(self,array1, array2, tolerance=1e-3):


        # 计算差值的绝对值
        arr=([abs(array2[i]-array1[i]) for i in range(len(array2))])
        # print(arr)
        absolute_difference = max(arr)

        print('两个数组差值最大为：',(absolute_difference))

        # 判断差值是否小于阈值
        if (absolute_difference < tolerance):
            return True
        else:
            return False


    def is_close_two_array(self,w1,ws,tolerance):
        for w in ws:
            if(self.compare_convergence(w,w1,tolerance)):
                return True
        return False

    # 过滤式特征选择
    def reliefF(self,T,tolerance,outputfile='../../out/exmp2/xor_w.csv'):

        data_w = []
        data_w.append(self.W.copy())
        for i in range(T):
            # self.classModel = EVM.get_model(self.get_nearmisdis_with_w(0.5), self.Y)
            print('\033[91m' + f'迭代次数为：{i}' + '\033[0m')
            # print('\033[91m' + f'w ：{self.W}' + '\033[0m')
            nearmisdis_model=self.get_nearmisdis_with_w()
            # nearmisdis=[n*2 for n in nearmisdis_model]
            # print(nearmisdis)
            self.classModel = EVM.get_model(nearmisdis_model, self.Y)

            self.weight = EVM.get_pro_with_model(self.classModel, nearmisdis_model)
            # print(self.weight)
            # print(self.weight)

            # self.weight = EVM.get_prob(self.get_nearmisdis_with_w(), self.Y)

            # self.weight = EVM.get_pro_with_model(self.classModel, self.get_nearmisdis_with_w())
            # print(sorted(self.get_nearmisdis_with_w()))
            # print(self.weight)

            self.getNNModel()

            w_t=self.get_metix_Z(self.init_w)

            print('\033[91m' + f'w_t ：{w_t}' + '\033[0m')
            if(self.compare_convergence(w_t,self.W.copy(),tolerance) or (min(self.weight)==1)):
                self.W = w_t
                print('算法收敛')
                print('\033[91m' + f'最终的子集挑选如下：{self.W}' + '\033[0m')
                return self.W

            if(i==T-1):
                print('\033[91m' + f'算法收敛失败，采用第一次的结果进行，w的最终结果为：{self.W}' + '\033[0m')
                self.W = data_w[1] if len(data_w)>=2 else self.W
                return self.W
            self.W=w_t
            data_w.append(self.W.copy())

        utiles.array_to_csv(outputfile, data_w)

        return self.W

    def get_final(self, cooling_rate):

        # print('\033[91m' + '******************进入特征选择环节*******************' + '\033[0m')
        #
        # old_margin = float('-inf')
        #
        # w = self.W.copy()
        # length = len(w)
        #
        # print('特征数量的大小如下：', length)
        #
        # w_new = [0] * length
        #
        # w_finall = [0] * length
        #
        # self.oldmargin = []
        #
        # w_max_index = []
        #
        # self.temperature = self.initial_temperature
        #
        # # self.get_margin_with_w(w)
        # #
        # # self.NH=self.nearhit
        # # self.NM=self.nearmiss
        #
        # last_oldmargin = old_margin
        # count = 1
        # while (True):
        #
        #     w_copy = w.copy()
        #     # 计算数组的平均值
        #     # average = (sum(w_copy) / len(w_copy))*0.5
        #
        #     # w_copy=[-1 if (w_finall[i]==1) or ((count>1) and (w_copy[i]<average)) else w_copy[i] for i in range(len(w_copy))]
        #     w_copy = [-1 if (w_finall[i] == 1) else w_copy[i] for i in range(len(w_copy))]
        #
        #     print(w_copy)
        #     print('\033[91m' + f'**********************当前筛选子集的迭代次数为：{count}'+ '\033[0m')
        #     # 使用列表推导来筛选非-1的元素，并计算个数
        #     non_minus_ones_count = len([x for x in w_copy if x != -1])
        #
        #     for i in range(non_minus_ones_count):
        #         # 找到最大值的下标
        #         max_index = np.argmax(w_copy)
        #
        #         print('*************当前遍历的特征为：', max_index)
        #         print('\033[91m' + f'*************当前筛选进度：{(i + 1) / len(w)}'+ '\033[0m')
        #
        #         w_new[max_index] = 1
        #
        #         w_t = self.get_metix_Zwith_w(w_new)
        #         new_margin = self.get_objectFunction(w_new, w_t)
        #
        #         w_copy[max_index] = -1
        #
        #         ran = random.random()
        #
        #         deff = (new_margin - old_margin)
        #
        #         print('old margin:', old_margin)
        #         print('new margin:', new_margin)
        #         print('temp:', self.temperature)
        #         if deff < 0:
        #             print('*************************************************************')
        #             print('p is :', math.exp((new_margin - old_margin) / self.temperature))
        #             print('random is :', ran)
        #
        #         if (old_margin < new_margin or (
        #                 deff < 0 and ran < math.exp((new_margin - old_margin) / self.temperature))):
        #
        #             print('******************当前的特征能够让margin增加*******************')
        #             print('当前特征为：', max_index)
        #
        #             print('当前最优的子集：', w_max_index)
        #             w_max_index.append(max_index)
        #             print('加入了该特征后的最优特征子集为：', w_max_index)
        #
        #             old_margin = new_margin
        #
        #             w_finall[max_index] = 1
        #             print('当前特征为：', w_finall)
        #
        #             self.oldmargin.append(old_margin)
        #
        #         else:
        #             print('******************当前的特征无法让margin增加*******************')
        #             print('当前特征为：', max_index)
        #             w_new[max_index] = 0
        #             w_finall[max_index] = 0
        #
        #         # 降低温度
        #         self.temperature *= cooling_rate
        #
        #     cur_oldmargin = old_margin
        #     print('上次迭代的margin值大小：', last_oldmargin)
        #     print('当前迭代的margin值大小：', cur_oldmargin)
        #     if (abs(cur_oldmargin - last_oldmargin) <= 0.05):
        #         print('当前筛选出来的子集已经收敛，当前迭代次数为：', count)
        #         print('最后的子集为：', w_finall)
        #         break
        #     else:
        #         print('当前迭代选出的子集的margin值更优: ', w_finall)
        #
        #         count += 1
        #         last_oldmargin = cur_oldmargin
        #
        # print('最后的最优子集为：', w_max_index)
        #
        # if (len(w_max_index) == 1):
        #     print('当前的最优子集的长度过小')
        #     arr = self.W.copy()
        #     arr1 = self.W.copy()
        #
        #     val = utiles.find_max_gap_with_value(arr)
        #     w_finall = [1 if a > val else 0 for a in arr1]
        #
        # print('最后的特征选择结果为：', w_finall)

        return [1 if w>0 else 0 for w in self.W]


if __name__ == '__main__':
    a=0
    # x,y=utiles.get_scaler_data('../data/uci_data_w/solar-flare_2.data')
    # a=evrelief_plus1(x,y,1,1,5)
    #
    # a.reliefF(30,1e-06,'../out/exmp2/xor_w.csv')


    # a.reliefF(30,0.06,'../out/exmp2/xor_w.csv')
    # a.get_final(0.2)


    # a.reliefF(10,10,'../out/exmp2/xor_w.csv')
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

