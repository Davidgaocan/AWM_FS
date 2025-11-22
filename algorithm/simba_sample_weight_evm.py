
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
from config import config
from utiles import utiles

'''
适用于多分类问题
'''
class Simba_sample_weight_evm:
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

        self.initial_temperature = config.initial_temperature
        self.cooling_rate = config.cooling_rate
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

        nearest_same_class_sample_distance = weighted_diff[same_class_indices][argmin_near]

        argmin_diff = np.argmin(weighted_diff[different_class_indices])
        nearest_different_class_sample_index = different_class_indices[argmin_diff]
        nearest_different_class_sample_distance = weighted_diff[different_class_indices][argmin_diff]


        self.nearhit[i] = nearest_same_class_sample_index

        self.nearmiss[i] = nearest_different_class_sample_index

        self.nearhitdis[i] = round(nearest_same_class_sample_distance, 2)
        self.nearmissdis[i] = round(nearest_different_class_sample_distance, 2)

        self.margin[i] = (self.nearmissdis[i] - self.nearhitdis[i]) * (1 / 2)*self.weight[i]

        return nearest_same_class_sample_index, nearest_different_class_sample_index

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


    #获取初始的margin矩阵
    def get_margin(self):
        for i in range(len(self.X)):
            self.find_nearest_neighbors(i)
        self.margin_init=self.margin.copy()

        return sum(self.margin_init)

    # 获取初始的margin矩阵
    def get_margin_with_w(self,w):
        for i in range(len(self.X)):
            self.find_nearest_neighbors_with_W(i,w)
        self.margin_init = self.margin.copy()

        return sum(self.margin_init.values())

    # 过滤式特征选择
    def reliefF(self, outputfile='../../out/exmp2/xor_w.csv', sample_way_isSampleWeight=True):

        m, n = np.shape(self.X)  # m为行数，n为列数

        data_w = []
        data_w.append([1] * n)

        self.get_margin()
        # print(m)
        # print(len(np.where(np.array(list(self.margin.values()))<0)[0]))
        # sys.exit()
        # self.weight = EVM.get_prob(self.nearmissdis, self.Y)


        self.weight = EVM.get_prob(self.nearmissdis, self.Y)
        # weight=np.array(self.weight)
        # # print(weight)
        # mar=np.array(list(self.margin.values()))
        #
        # w=(weight[np.where(mar>0)[0]])
        # print(mar[np.where((mar<0)&(weight>0.95))])
        # sys.exit()
        # self.weight = utiles.kerneldensity(self.X, self.Y)


        # self.weight=[1-self.weight[i] if self.margin[i]<0 else self.weight[i] for i in range(len(self.weight))]

        # self.weight = [1]*m
        self.get_margin()
        old_margin=sum(self.margin.values())

        flag=False
        t_while=0

        max_margin=float('-inf')
        max_w=self.W.copy()
        t_flag=0

        self.marginadd = []

        isClose=False
        t_isClose=0
        close=0.01
        self.marginadd.append(old_margin)

        while(True):

            print('t:', t_while)


            w_range=[0 for f in self.X.columns]

            # 判断是否是按照样本权重采样
            if sample_way_isSampleWeight:
                # 如果按照样本权重采样，则通过赋予不同位置上的权重。然后按照权重重复采样
                # sample_index = utiles.get_sample_with_weigth(self.weight, m, self.__sample_num)

                sample_index = random.sample(range(0, m), self.__sample_num)
            else:
                # 否则按照随机取样方法进行

                sample_index = random.sample(range(0, m), self.__sample_num)


            # print(sample_index)

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
                    # margin_init=1 if self.margin[i] ==0 else self.margin[i]
                    # print(self.margin[i])
                    limba = 0
                    if self.margin[i] <= 0:
                        # limba = 0
                        limba = (np.exp(self.margin[i] * (-1)))
                    else:
                        limba = (np.exp(self.margin[i] * (-1)))

                    w_range[f_index] += round(self.e[i] * (self.W[f_index]) * limba * self.weight[i], 2)

            # print(self.W)
            w_copy=[w for w in self.W]
            # print(w_range)
            W_new=[w_copy[i]+w_range[i] for i in range(len(w_copy))]

            # print(W_new)
            # print(self.W)

            W_new_norm = self.get_norm(W_new.copy())
            new_margin = self.get_margin_with_w(W_new_norm)

            if np.isnan(new_margin):
                break
            # new_margin=self.get_margin_with_w(W_new) if not abs(self.get_margin_with_w(W_new) ) > sys.float_info.max else old_margin


            if new_margin>max_margin :
                max_margin=new_margin
                max_w=W_new

            # print(W_new)
            # print(self.W)
            print('old margin:', old_margin)
            print('new margin:', new_margin)

            if ((t_while > 50) or (t_isClose >= 5 and isClose)):

                print('终止条件：')
                if (t_isClose >= 5):
                    print('margin值在趋于收敛后迭代5次后未发现有提升情况')
                else:
                    print('迭代次数超过：', t_while)
                    if (not flag):
                        print('未能找到很好的w，能够使得其margin大于最初的w')
                        self.W = max_w
                    else:
                        print('找到当前最大的margin值，最大margin值在循环超过次数后还是没有变化')

                data_w.append(self.get_norm(self.W.copy()))
                break


            elif((old_margin-new_margin)<(-close)):
                print('old<new')
                self.marginadd.append(old_margin)
                t_while = 0

                if isClose:
                    t_isClose=0
                    isClose=False


                t_flag += 1
                flag = True
                old_margin = new_margin
                self.W = W_new
                data_w.append(self.get_norm(self.W.copy()))

            elif (old_margin - new_margin>close):
                print('old>new')

                if isClose:
                    t_isClose += 1
                    self.marginadd.append(old_margin)

                t_while += 1





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





        utiles.array_to_csv(outputfile, data_w)

        # print('采样各样本特征权重如下：')
        # print(self.W)
        # print('平均特征权重如下：')
        self.W = self.get_norm(self.W.copy())
        print('     w         is   :    ',self.W)

        return self.W


    # 过滤式特征选择
    # def reliefF(self,outputfile='../out/exmp2/xor_w.csv',sample_way_isSampleWeight=True):
    #
    #     # print sample
    #     m, n = np.shape(self.X)  # m为行数，n为列数
    #
    #     data_w = []
    #     data_w.append([1]*n)
    #
    #     self.get_margin()
    #     #
    #     # self.weight=utiles.margin_weight(1,np.array(list(self.margin_init.values())))
    #     # self.weight=utiles.GaussianMixtureToProb_plus(self.X,self.Y)
    #     # self.weight=utiles.kerneldensity(self.X,self.Y)
    #     self.weight=EVM.get_prob(self.nearmissdis,self.Y)
    #
    #
    #
    #     print(self.weight)
    #     # sys.exit()
    #
    #     # self.weight=utiles.min_max_normalize(self.weight.copy())
    #     # print(self.weight)
    #     # print(self.weight)
    #     # print(max(self.weight))
    #     # print(min(self.weight))
    #     # index=np.where(self.weight>0.3)
    #     # print(index)
    #     # features=self.X
    #     # # 绘制概率较高的数据分布
    #     # plt.scatter(features.iloc[:, 0], features.iloc[:, 1], color='blue',
    #     #             label='High Probability')
    #     # plt.scatter(features.iloc[index[0], 0], features.iloc[index[0], 1], color='red',
    #     #             label='Low Probability')
    #     # plt.xlabel('Feature 1')
    #     # plt.ylabel('Feature 2')
    #     # plt.title('Data Distribution with High Probability')
    #     # plt.legend()
    #     # plt.show()
    #     # sys.exit()
    #
    #     #判断是否是按照样本权重采样
    #     if sample_way_isSampleWeight:
    #         # 如果按照样本权重采样，则通过赋予不同位置上的权重。然后按照权重重复采样
    #         sample_index=utiles.get_sample_with_weigth(self.weight,m,self.__sample_num)
    #         # self.weight=(weight[i] for i in index)
    #     else:
    #         #否则按照随机取样方法进行
    #         sample_index=random.sample(range(0, m), self.__sample_num)
    #
    #
    #
    #
    #     for i in sample_index:  # 采样次数
    #
    #         try:
    #             NearHit, NearMiss = self.find_nearest_neighbors(i)
    #
    #         except:
    #             continue
    #
    #         # print('第 %s 次采样，样本index为 %s，其NearHit k近邻行索引为 %s ，NearMiss k近邻行索引为 %s' % (num, i, NearHit, NearMiss))
    #         for f in self.X.columns:
    #             # print('***:', f, i, NearHit, NearMiss)
    #
    #             self.get_weight(f, i)
    #
    #             f_index = self.X.columns.get_loc(f)
    #             # margin_init=1 if self.margin[i] ==0 else self.margin[i]
    #             # print(self.margin[i])
    #             limba=0
    #             if self.margin[i]<=0:
    #                 limba=0
    #             else:
    #                 limba=(np.exp(self.weight[i] * self.margin[i]*(-1)))
    #
    #
    #             w_range = round(self.e[i] * (self.W[f_index])*limba*self.weight[i], 2)
    #             self.W[f_index] += w_range
    #
    #             # print('特征 %s 的权重变化为 %s.' % (f, w_range))
    #         # score.append(one_score)
    #
    #         data_w.append(self.get_norm(self.W.copy()))
    #
    #     utiles.array_to_csv(outputfile,data_w)
    #
    #     print('采样各样本特征权重如下：')
    #     print(self.W)
    #     print('平均特征权重如下：')
    #     self.W = self.get_norm(self.W.copy())
    #     print(self.W)
    #
    #     return self.W




    def get_norm(self, W):
        # W = [0 if w < 0 else w for w in W]
        w_squared = np.square(W)

        w_norm_inf = np.linalg.norm(w_squared, ord=np.inf)

        normalized_w = w_squared / w_norm_inf

        return normalized_w

        # 返回最终选取的特征

    def get_final(self):
        print('******************进入特征选择环节*******************')
        print('**************************************************')

        # f_w = pd.DataFrame(self.W, columns=['weight'])
        # print(f_w)
        # m, n = np.shape(self.__data)
        # final_feature_t = f_w[f_w['weight'] > self.__t]

        #寻找当前最大的w，然后选取当前的特征，然后直到找到最大margin的特征组合：

        old_margin=float('-inf')

        w=self.W.copy()
        print('特征数量的大小如下：', len(w))
        w_new = [0] * len(w)


        w_finall=[0] * len(w)


        self.oldmargin=[]

        w_max_index=[]

        self.temperature=self.initial_temperature


        for i in range(len(w)):
            # 找到最大值的下标
            max_index = np.argmax(w)

            # 找到最大值
            max_value = w[max_index]

            print('*************当前遍历的特征为：',max_index)
            print('*************当前筛选进度：',(i+1)/len(w))



            w_new[max_index] = max_value

            new_margin = self.get_margin_with_w(w_new)

            w[max_index] = -1

            ran=random.random()

            deff=(new_margin-old_margin)

            print('old margin:', old_margin)
            print('new margin:', new_margin)
            print('temp:', self.temperature)
            if deff < 0:
                print('*************************************************************')
                print('p is :', math.exp((new_margin - old_margin) / self.temperature))
                print('random is :', ran)


            if(old_margin<new_margin or  (deff < 0 and ran< math.exp( (new_margin-old_margin)/ self.temperature))):



                print('******************当前的特征能够让margin增加*******************')
                print('当前特征为：',max_index)



                print('当前最优的子集：',w_max_index)
                w_max_index.append(max_index)
                print('加入了该特征后的最优特征子集为：',w_max_index)

                old_margin=new_margin

                w_finall[max_index]=1

                self.oldmargin.append(old_margin)

            else:
                print('******************当前的特征无法让margin增加*******************')
                print('当前特征为：', max_index)
                w_new[max_index]=0
                w_finall[max_index] = 0

            # 降低温度
            self.temperature *= self.cooling_rate

        print('最后的最优子集为：', w_max_index)

        if(len(w_max_index)==1):
            print('当前的最优子集的长度过小')
            arr=self.W.copy()
            arr1=self.W.copy()

            val=utiles.find_max_gap_with_value(arr)
            w_finall=[1 if a>val else 0 for a in arr1]

        print('最后的特征选择结果为：',w_finall)





        return w_finall


if __name__ == '__main__':
    a=0
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

