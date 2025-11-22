import sys

import pandas as pd
import numpy as np
import numpy.linalg as la
import random
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# from algorithm.Nsga2_simba import NSGA2_simba
from utiles import utiles

'''
适用于多分类问题
'''


class relief_plus:
    def __init__(self, X, Y, sample_rate, t, k):
        """
        #
        :param data_df: 数据框（字段为特征，行为样本）
        :param sample_rate: 抽样比例
        :param t: 统计量分量阈值
        :param k: k近邻的个数
        """
        self.X = X

        self.Y = Y

        self.X = self.process_discrete_features(self.X)
        m, n = np.shape(self.X)  # m为行数，n为列数
        self.w=[0]*n


        # print(self.__feature[:-1])
        # sys.exit()
        self.__sample_num = int(round(len(self.X) * sample_rate))
        self.__t = t
        self.__k = k

        # 数据处理（将离散型数据处理成连续型数据，比如字符到数值）
    def process_discrete_features(selif, x):
        new_dataframe = x.copy()
        label_encoder = LabelEncoder()
        for column in new_dataframe.columns:
            if new_dataframe[column].dtype == 'object':
                new_dataframe[column] = label_encoder.fit_transform(new_dataframe[column])

        return new_dataframe

    def find_nearest_neighbors_plus(self, i,w):

        X = self.X
        Y = self.Y
        weights = w
        weighted_diff = np.linalg.norm(weights * (X - X.iloc[i]), axis=1)

        same_class_indices = np.where(Y == Y.iloc[i])[0]
        same_class_indices = same_class_indices[same_class_indices != i]
        different_class_indices = np.where(Y != Y.iloc[i])[0]

        argmin_near = np.argsort(weighted_diff[same_class_indices])[:self.__k]
        nearest_same_class_sample_index = same_class_indices[argmin_near]
        # nearest_different_class_sample_index={}

        argmin_diff = np.argsort(weighted_diff[different_class_indices])[:self.__k]
        nearest_different_class_sample_index = different_class_indices[argmin_diff]


        return nearest_same_class_sample_index, nearest_different_class_sample_index


    def getProb(self,i):
        Y=self.Y

        same_class_indices = np.where(Y == Y.iloc[i])[0]

        return len(same_class_indices)/len(Y)




    # 计算特征权重
    def get_weight(self, feature, index, NearHit, NearMiss):

        maxvalue=max(self.X.iloc[:,feature])
        minvalue=min(self.X.iloc[:,feature])

        nh=[abs(self.X.iloc[index,feature]-self.X.iloc[h,feature]) for h in NearHit]
        # print(maxvalue)
        # print(minvalue)
        # print(maxvalue==minvalue)
        if  maxvalue==minvalue:
            k_nh=0
        else :
            # print(((maxvalue - minvalue) * self.__sample_num *len(nh)))
            k_nh = (1 / ((maxvalue - minvalue) * self.__sample_num *len(nh)))

        sum_nh=k_nh*sum(nh)
        pi=self.getProb(index)

        nm=[abs((self.X.iloc[index,feature]-self.X.iloc[h,feature] ))*(self.getProb(h)/(1-pi)) for h in NearMiss]
        if  maxvalue==minvalue:
            k_nm=0
        else :
            # print(((maxvalue - minvalue) * self.__sample_num *len(nh)))
            k_nm = 1 / ((maxvalue - minvalue) * self.__sample_num * len(nm))
        # k_nm = 1 / ((maxvalue - minvalue) * self.__sample_num * len(nm))
        sum_nm=sum(nm)*k_nm

        return sum_nm-sum_nh

    # 过滤式特征选择
    def reliefF(self):
        sample = self.X
        # print sample
        m, n = np.shape(self.X)  # m为行数，n为列数
        score = []
        sample_index = random.sample(range(0, m), self.__sample_num)
        # print('采样样本索引为 %s ' % sample_index)
        w=[1]*n

        for i in sample_index:  # 采样次数

            NearHit, NearMiss = self.find_nearest_neighbors_plus(i,w)
            # print('第 %s 次采样，样本index为 %s，其NearHit k近邻行索引为 %s ，NearMiss k近邻行索引为 %s' % (num, i, NearHit, NearMiss))

            for f in self.X.columns:
                # print('***:',f,i,NearHit,NearMiss)
                w_value = self.get_weight(f, i, NearHit, NearMiss)
                f_index = self.X.columns.get_loc(f)
                self.w[f_index]+=w_value
                # print('特征 %s 的权重为 %s.' % (f, w))



        print('平均特征权重如下：',self.w)


        return self.w

    def get_final(self, cooling_rate):
        print('******************进入特征选择环节*******************')
        print('**************************************************')
        x = self.X
        y = self.Y

        # a = NSGA2_simba(10, 100, len(self.w), self.w.copy(), x, y)
        final = a.get_subset()

        return final[0]


# 几种距离求解

# 欧氏距离(Euclidean Distance)
def eulidSim(vecA, vecB):
    return la.norm(vecA - vecB)


# 余弦相似度
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


# 皮尔逊(皮尔森)相关系数
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
        return 0.5 + 0.5 * np.corrcoef(vecA, vecB, rowvar=0)[0][1]

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

if __name__ == '__main__':
    a = 0
    X_train, X_test, y_train, y_test = get_scaler_data('../data/uci_data_w/iris.data')


    a = relief_plus(X_train, y_train, 1, 1, 1)
    a.reliefF()
    # a.reliefF()
    # with open('./西瓜数据集30.csv','r',encoding= 'gbk') as f:
    #     data = pd.read_csv(f)[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率', '好瓜']]
    #     #print(type(data))
    #
    #     # print(data)
    #     # f_csv = csv.reader(f)
    #     # for row in f_csv:
    #     #     print(row)
    #     f = Relief(data, 1, 0.2, 1)
    #     # df = f.get_data()
    #     # print(type(df.iloc[0]))
    #     # f.get_neighbors(df.iloc[0])
    #     f.reliefF()
    #     f.get_final()
    #
