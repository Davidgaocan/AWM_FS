
import math
import sys

import pandas as pd
import numpy as np
import numpy.linalg as la
import random
import csv

from simba import Relief
from utiles import utiles

'''
适用于多分类问题
'''
class Simba:
    def __init__(self, data_df, sample_rate, t, k):
        """
        #
        :param data_df: 数据框（字段为特征，行为样本）
        :param sample_rate: 抽样比例
        :param t: 统计量分量阈值
        :param k: k近邻的个数
        """
        self.__data = data_df
        self.__feature = data_df.columns

        self.__data = self.get_data()
        self.__sample_num = int(round(len(data_df) * sample_rate))
        self.__t = t
        self.__k = k
        self.W=[1]*(len(data_df.columns)-1)
    # 数据处理（将离散型数据处理成连续型数据，比如字符到数值）
    def get_data(self):
        new_data = pd.DataFrame()

        for one in self.__feature[:-1]:
            col = self.__data[one]
            if (str(list(col)[0]).split(".")[0]).isdigit() or str(list(col)[0]).isdigit() or (str(list(col)[0]).split('-')[-1]).split(".")[-1].isdigit():

                new_data[one] = self.__data[one]
                # print('%s 是数值型' % one)
            else:
                # print('%s 是离散型' % one)
                keys = list(set(list(col)))
                values = list(range(len(keys)))
                new = dict(zip(keys, values))
                new_data[one] = self.__data[one].map(new)
        new_data[self.__feature[-1]] = self.__data[self.__feature[-1]]
        return new_data

    # 返回一个样本的k个猜中近邻和其他类的k个猜错近邻
    def get_neighbors(self, row):
        df = self.__data
        row_type = row[df.columns[-1]]
        right_df = df[df[df.columns[-1]] == row_type].drop(columns=[df.columns[-1]])
        aim = row.drop(df.columns[-1])
        # f = lambda x: eulidSim(np.mat(x), np.mat(aim))
        f = lambda x: eulidSim(np.mat(x.values), np.mat(aim.values))
        right_sim = right_df.apply(f, axis=1)
        # print(row)
        right_sim_two = right_sim.drop(right_sim.idxmin())
        right = dict()
        right[row_type] = list(right_sim_two.sort_values().index[0:self.__k])
        # print list(right_sim_two.sort_values().index[0:self.__k])
        lst = [row_type]
        types = list(set(df[df.columns[-1]]) - set(lst))
        wrong = dict()
        for one in types:
            wrong_df = df[df[df.columns[-1]] == one].drop(columns=[df.columns[-1]])
            wrong_sim = wrong_df.apply(f, axis=1)
            wrong[one] = list(wrong_sim.sort_values().index[0:self.__k])
        # print(right, wrong)
        return right, wrong

    # 计算特征权重
    def get_weight(self, feature, index, NearHit, NearMiss):
        # data = self.__dat
        # a.drop(self.__feature[-1], axis=1)
        data = self.__data
        row = data.iloc[index]
        right = 0

        for one in list(NearHit.values())[0]:
            nearhit = data.iloc[one]
            if (str(row[feature]).split(".")[0]).isdigit() or str(row[feature]).isdigit() or (str(row[feature]).split('-')[-1]).split(".")[-1].isdigit():
                max_feature = data[feature].max()
                min_feature = data[feature].min()
                right_one = pow(round(abs(row[feature] - nearhit[feature]) / (max_feature - min_feature), 2), 2)
            else:

                right_one = 0 if row[feature] == nearhit[feature] else 1
            right += right_one
        right_w = round(right / self.__k, 2)

        wrong_w = 0
        # 样本row所在的种类占样本集的比例
        p_row = round(float(list(data[data.columns[-1]]).count(row[data.columns[-1]])) / len(data), 2)
        for one in NearMiss.keys():
            # 种类one在样本集中所占的比例
            p_one = round(float(list(data[data.columns[-1]]).count(one)) / len(data), 2)
            wrong_one = 0
            for i in NearMiss[one]:
                nearmiss = data.iloc[i]
                if (str(row[feature]).split(".")[0]).isdigit() or str(row[feature]).isdigit() or (str(row[feature]).split('-')[-1]).split(".")[-1].isdigit():
                    max_feature = data[feature].max()
                    min_feature = data[feature].min()
                    wrong_one_one = pow(round(abs(row[feature] - nearmiss[feature]) / (max_feature - min_feature), 2), 2)
                else:
                    wrong_one_one = 0 if row[feature] == nearmiss[feature] else 1
                wrong_one += wrong_one_one

            wrong = round(p_one / (1 - p_row) * wrong_one / self.__k, 2)
            wrong_w += wrong
        w = wrong_w - right_w
        return right_w,wrong_w

    def cal_e(self, row, NearHit, NearMiss,W):
        # F代表特征的下标


        row=row[:-1]

        NearHit_data=self.__data.iloc[list(NearHit.values())[0]]

        NearHit_data=NearHit_data.drop(columns=[NearHit_data.columns[-1]])
        # print((NearHit_data))
        # sys.exit()
        NearMiss_data=self.__data.iloc[list(NearMiss.values())[0]]
        NearMiss_data = NearMiss_data.drop(columns=[NearMiss_data.columns[-1]])
        # print((row.values))
        f = lambda x: eulidSim_W((x.values),(row.values),W)

        right_sim = NearHit_data.apply(f, axis=1)
        right_sim=(right_sim).mean()

        wrong_sim = NearMiss_data.apply(f, axis=1)

        wrong_sim = (wrong_sim).mean()
        return right_sim,wrong_sim

    # 过滤式特征选择
    def reliefF(self):
        sample = self.__data
        # print sample
        m, n = np.shape(self.__data)  # m为行数，n为列数
        data_w = []
        # data_w.append(self.W.copy())

        sample_index = random.sample(range(0, m), self.__sample_num)
        # print('采样样本索引为 %s ' % sample_index)
        num = 1
        # index=utiles.get_rbf_mid_index()
        for i in sample_index:    # 采样次数
            # one_score = dict()
            row = sample.iloc[i]
            NearHit, NearMiss = self.get_neighbors(row)
            # print('第 %s 次采样，样本index为 %s，其NearHit k近邻行索引为 %s ，NearMiss k近邻行索引为 %s' % (num, i, NearHit, NearMiss))
            for f in self.__feature[0:-1]:

                right_w, wrong_w= self.get_weight(f, i, NearHit, NearMiss)
                # print(right_w,wrong_w)

                right_w1, wrong_w1= self.cal_e(row, NearHit, NearMiss,self.W)

                # print(right_w1, wrong_w1)
                # print(((wrong_w/wrong_w1)-(right_w/right_w1))*(1/2))
                # sys.exit()

                f_index=self.__feature[0:-1].get_loc(f)
                # print(right_w1,wrong_w1)
                # sys.exit()
                fronter=0 if (wrong_w1==0) else (wrong_w/wrong_w1)
                laster=0 if (right_w1==0) else (right_w/right_w1)

                w_range=round(((fronter)-(laster))*(1/2)*(self.W[f_index]),2)

                self.W[f_index] += w_range

            # score.append(one_score)
            num += 1
            data_w.append(self.get_norm(self.W.copy()))
            # print(data_w)
            # sys.exit()
            # print(self.W)
            # sys.exit()
        # f_w = pd.DataFrame(score)


        # utiles.array_to_csv('./out/rbf/class0_100_mid.csv',data_w,2)

        self.W=self.get_norm(self.W.copy())

        # self.W = np.square(self.W)
        # # 计算向量的模
        # norm = np.linalg.norm(self.W)
        #
        # # 进行归一化操作
        # self.W = self.W / norm
        # print('采样各样本特征权重如下：')
        # print(self.W)
        return self.W

    # 返回最终选取的特征
    def get_norm(self,W):
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

    def feature_ranking(self):
        idx = np.argsort(self.W, 0)
        return idx[::-1]

# 几种距离求解

#欧氏距离(Euclidean Distance)
def eulidSim(vecA, vecB):
    return la.norm(vecA - vecB)

#欧氏距离(Euclidean Distance)
def eulidSim_W(point1, point2,W):
    # print((point1))
    # print((point2))
    # print(len(W))
    # sys.exit()

    if len(point1) != len(point2) or len(point1) != len(W):
        raise ValueError("Points and weight vector must have the same dimensions.")

    squared_sum = 0
    for i in range(len(point1)):
        pow_num=(math.pow(W[i],2)*math.pow(point1[i] - point2[i],2))
        squared_sum += pow_num
    distance = math.sqrt(squared_sum)
    return distance

#余弦相似度
def cosSim(vecA, vecB):
    """
    :param vecA: 行向量
    :param vecB: 行向量
    :return: 返回余弦相似度（范围在0-1之间）
    """
    num = float(vecA * vecB.T)
    denom = la.norm(vecA) * la.norm(vecB)
    cosSim = 0.5 + 0.5 * (num / denom)
    return cosSim

#皮尔逊(皮尔森)相关系数
'''
皮尔森相关系数也称皮尔森积矩相关系数(Pearson product-moment correlation coefficient) ，
是一种线性相关系数，
是最常用的一种相关系数。
记为r，用来反映两个变量X和Y的线性相关程度，
r值介于-1到1之间，绝对值越大表明相关性越强。
'''
def pearsSim(vecA, vecB):
    if len(vecA) < 3:
        return 1.0
    else:
        return 0.5 + 0.5 * np.corrcoef(vecA, vecB,rowvar=0)[0][1]

def get_data(data):
    new_data = pd.DataFrame()
    __feature = data.columns

    for one in __feature[:]:
        col = data[one]
        if (str(list(col)[0]).split(".")[0]).isdigit() or str(list(col)[0]).isdigit() or \
                (str(list(col)[0]).split('-')[-1]).split(".")[-1].isdigit():
            new_data[one] = data[one]
            # print('%s 是数值型' % one)
        else:
            # print('%s 是离散型' % one)
            keys = list(set(list(col)))
            values = list(range(len(keys)))
            new = dict(zip(keys, values))
            new_data[one] = data[one].map(new)

    return new_data
if __name__ == '__main__':
    a=0
#     # with open('./西瓜数据集30.csv','r',encoding= 'gbk') as f:
#     #     data = pd.read_csv(f)[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率', '好瓜']]
    data = pd.read_csv('../data/data_all_final/rbf_data.data', encoding="gbk")

    data = get_data(data)

    f = Simba(data, 1, 0.2, 1)
    df = f.get_data()

    w = f.reliefF()
    print(w)
#     w_t = f.get_final()
#
#     # w=w.reshape(1,2)
#     # print(w)
#     # sys.exit()
#     # utiles.array_to_csv('./out/rbf/class0_100_bound.csv',w,2)
#     # print(w)
#     # fea = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']
#     # # 将数组拼接成DataFrame
#     # df = pd.DataFrame({'Column1': fea, 'Column2': w})
#
#     # 保存DataFrame到CSV文件
#     # df.to_csv('./out/simba.csv', index=False)
#     # f.get_final()

