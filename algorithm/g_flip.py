import math
import sys
import time

import pandas as pd
import numpy as np
import numpy.linalg as la
import random
class FilterError:
    pass


class G_flip:
    def __init__(self, data_df,label,k):
        """
        #
        :param data_df: 数据框（字段为特征，行为样本）
        :param sample_rate: 抽样比例
        :param t: 统计量分量阈值
        :param k: 选取的特征的个数
        """
        self.data = data_df
        self.label = label
        self.feature = data_df.columns
        self.num_sample, num_featuren = np.shape(self.data)
        self.num_featuren=num_featuren -1
        self.k=k
    def getRandFeaturenIndex(self):
        num_featuren=self.num_featuren
        all_features = np.arange(num_featuren)  # 获取所有特征
        # print("All features:", all_features)

        np.random.shuffle(all_features)  # 对特征进行重新排列
        return all_features

        # 返回一个样本的k个猜中近邻和其他类的k个猜错近邻
        # 注意这里的row代表的是已经传入进去的随机特征后的row

    def get_neighbors(self, row):
        df = self.data
        row_type = row[df.columns[-1]]
        right_df = df[df[df.columns[-1]] == row_type].drop(columns=[df.columns[-1]])
        aim = row.drop(df.columns[-1])
        # f = lambda x: eulidSim(np.mat(x), np.mat(aim))
        f = lambda x: eulidSim(np.mat(x.values), np.mat(aim.values))

        right_sim = right_df.apply(f, axis=1)
        # print(right_sim)
        # print(right_sim.sum())
        # print((right_sim).mean())
        sys.exit()
        right_sim_two = right_sim.drop(right_sim.idxmin())
        right = dict()
        right[row_type] = list(right_sim_two.sort_values().index[0:self.k])
        # print list(right_sim_two.sort_values().index[0:self.__k])
        lst = [row_type]
        types = list(set(df[df.columns[-1]]) - set(lst))
        wrong = dict()
        for one in types:
            wrong_df = df[df[df.columns[-1]] == one].drop(columns=[df.columns[-1]])
            wrong_sim = wrong_df.apply(f, axis=1)
            wrong[one] = list(wrong_sim.sort_values().index[0:self.k])
        # print(right, wrong)
        return right, wrong

    def get_neighbors_feature(self, feature,row,index):
        df = self.data
        row_type = row[df.columns[-1]]
        right_df = df[df[df.columns[-1]] == row_type].drop(columns=[df.columns[-1]])
        right_df =right_df[feature]
        aim = (row.drop(df.columns[-1]))[feature]


        f = lambda x: eulidSim(np.mat(x.values), np.mat(aim.values))
        right_sim = right_df.apply(f, axis=1)
        # print(right_sim.idxmin())
        right_sim_two = right_sim.drop(index)

        right = dict()
        right[row_type] = list(right_sim_two.sort_values().index[0:self.k])

        # print list(right_sim_two.sort_values().index[0:self.__k])
        lst = [row_type]
        types = list(set(df[df.columns[-1]]) - set(lst))
        wrong = dict()
        for one in types:
            wrong_df = df[df[df.columns[-1]] == one].drop(columns=[df.columns[-1]])
            wrong_df = wrong_df[feature]
            wrong_sim = wrong_df.apply(f, axis=1)
            wrong[one] = list(wrong_sim.sort_values().index[0:self.k])
        # print(right, wrong)
        return right, wrong

    def comput_e(self,features, index, NearHit, NearMiss):
        # F代表特征的下标
        data = self.data.drop(self.feature[-1], axis=1)
        row = data.iloc[index]
        right_w =0
        for one in list(NearHit.values())[0]:
            right = 0
            nearhit = data.iloc[one]
            for feature in features:
                if (str(row[feature]).split(".")[0]).isdigit() or str(row[feature]).isdigit() or \
                        (str(row[feature]).split('-')[-1]).split(".")[-1].isdigit():
                    max_feature = data[feature].max()
                    min_feature = data[feature].min()
                    right_one = pow(round(abs(row[feature] - nearhit[feature]) / (max_feature - min_feature), 2), 2)
                else:
                    # print('@@:', row[feature])
                    # print('$$:', nearhit[feature])
                    # print('-' * 100)
                    right_one = 0 if row[feature] == nearhit[feature] else 1
                right += right_one
            right_w+=round(math.sqrt(right), 2)
        right_w = round(right_w / self.k, 2)

        wrong_w = 0
        # 样本row所在的种类占样本集的比例
        p_row = round(float(list(data[data.columns[-1]]).count(row[data.columns[-1]])) / len(data), 2)
        for one in NearMiss.keys():
            # 种类one在样本集中所占的比例
            p_one = round(float(list(data[data.columns[-1]]).count(one)) / len(data), 2)
            wrong = 0
            for i in NearMiss[one]:
                wrong_one=0
                nearmiss = data.iloc[i]
                for feature in features:
                    if (str(row[feature]).split(".")[0]).isdigit() or str(row[feature]).isdigit() or \
                            (str(row[feature]).split('-')[-1]).split(".")[-1].isdigit():
                        max_feature = data[feature].max()
                        min_feature = data[feature].min()
                        wrong_one_one = pow(
                            round(abs(row[feature] - nearmiss[feature]) / (max_feature - min_feature), 2),
                            2)
                    else:
                        wrong_one_one = 0 if row[feature] == nearmiss[feature] else 1
                    wrong_one += wrong_one_one #求出一个点的距离
                wrong+=round(math.sqrt(wrong_one), 2)
            wrong = round(p_one / (1 - p_row) * wrong / self.k, 2)
            wrong_w += wrong
        # wrong_w =round(math.sqrt(wrong_w), 2)
        w = wrong_w - right_w
        return w


    def cal_e(self, row, NearHit, NearMiss):
        # F代表特征的下标

        NearHit_data=self.data.iloc[list(NearHit.values())[0]]

        NearMiss_data=self.data.iloc[list(NearMiss.values())[0]]

        f = lambda x: eulidSim(np.mat(x.values), np.mat(row.values))
        right_sim = NearHit_data.apply(f, axis=1)
        right_sim=(right_sim).mean()
        wrong_sim = NearMiss_data.apply(f, axis=1)

        wrong_sim = (wrong_sim).mean()
        return right_sim,wrong_sim

    def g_flip(self):
        df=self.data
        sampled_df = df.sample(frac=0.01, random_state=42)

        # print(sampled_df)
        # sys.exit()
        f=[264, 271, 265, 292, 345, 318, 94, 413, 252, 207, 409, 126, 491, 471, 69, 400, 86, 53, 302, 167, 420, 474, 166, 7, 146, 394, 332, 392, 357, 423, 451, 247, 473, 173, 402, 205]
        f_old=f.copy()
        e_old=round(float('-inf'),2)
        ite=0
        start_time = time.time()
        while(True):

            s=self.getRandFeaturenIndex()
            print("一次特征选择开始*******************")
            print("重新开始排序属性",s)
            print("目前最优子集",f_old)
            fea_num=0
            for si in s:
                fnew=f.copy()
                if (si in f):
                    fnew.remove(si)
                else :
                    fnew.append(si)
                print("当前属性：",si)
                print("当前f集合：",f)
                print("当前f_new属性：",fnew)

                features = self.feature[fnew]
                # print(features)
                # print(data[features])
                # sys.exit()
                e_new = 0
                fea_num+=1
                for i in sampled_df.index:
                    row=self.data.iloc[i]
                    NearHit, NearMiss=self.get_neighbors_feature(features,row,i)
                    # End the timer
                    # end_time = time.time()
                    # Calculate the elapsed time
                    # print(i)
                    # print(NearHit)
                    # print("***********")
                    # print(NearMiss)
                    # sys.exit()
                    right_sim,wrong_sim=self.cal_e(row,NearHit,NearMiss)

                    e_new +=(wrong_sim-right_sim)

                print ("修改特征集合后的e_new为：",round((e_new),2))

                print("最优的e为e_old:",e_old)

                if(e_new>=e_old):
                    e_old=e_new
                    f=fnew.copy()

                print('*******'+str(ite+1)+"次选择结束后"+'*********')
                print('*******'+str(fea_num)+"个特征"+'*********')

                print("f_old:",f_old)
                print("f:",f)
                print("e_old:",e_old)

            if set(f_old)==set(f):
                print("总的运行迭代次数为 ",ite)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print('时间为：' + str(elapsed_time))
                # sys.exit()
                return self.feature[f],round(e_old,2)
                break
            else:

                f_old=f.copy()
                ite+=1
                if ite>5:
                    print("总的运行迭代次数为 ", ite)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print('时间为：' + str(elapsed_time))
                    return self.feature[f], round(e_old, 2)
                    break
                print("当前迭代次数为 ", ite)
                print(f)
                print("当前最优特征 ", f)
                continue


#欧氏距离(Euclidean Distance)
def eulidSim(vecA, vecB):
    return la.norm(vecA - vecB)

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

# 数据处理（将离散型数据处理成连续型数据，比如字符到数值）
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
    # with open('./西瓜数据集30.csv','r',encoding= 'gbk') as f:
    #     data = pd.read_csv(f)[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率', '好瓜']]
    data = pd.read_csv('../data/madelon/train.csv', encoding="gbk")
    # 数据处理
    data=get_data(data)
    g_flip=G_flip(data,data.iloc[:,-1],1)

    # t=1
    f, e = g_flip.g_flip()
    print("**********************")
    print(f)
    print(e)

    # # sys.exit()
    # ew_old=round(float('-inf'),2)
    # while(t<=20):
    #     f,enew=g_flip.g_flip()
    #     if(ew_old==enew):
    #         print("****************结束****************************")
    #         print(ew_old)
    #         print(enew)
    #         break
    #     elif(ew_old<enew):
    #         print("****************一次最优****************************")
    #         print(ew_old)
    #         print(enew)
    #         ew_old = enew
    #
    #         df = pd.DataFrame({'Column1': f, })
    #         df.to_csv('./out/gflip/_' + str(ew_old) + '_.csv', index=False)
    #         t+=1
    #
    #         continue
    #     else:
    #         continue





    # print(g_flip.g_flip())
    # print(g_flip.getRandFeaturenIndex())
    # print(g_flip.data['根蒂'][0])
    # # print(g_flip.comput_e())
    # print(g_flip.feature)