import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from deap import base, creator, tools

from utiles import utiles


class NSGA2:
    def __init__(self, maxFEs, N, D, sc, x,y,k_fold=5):
        self.maxFEs = maxFEs
        self.N = N
        self.D = D
        self.sc = sc
        self.k_fold = k_fold
        self.population = self.initialize_population()
        self.currentFEs = 0
        self.toolbox = self.create_toolbox()
        self.P = self.toolbox.Population(n=self.N)
        self.x=x
        self.y=y

    def create_toolbox(self):

        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)



        toolbox = base.Toolbox()

        toolbox.register("Indices", self.generate_individual)
        toolbox.register("Getone", self.get_oi)

        toolbox.register("Individual", tools.initIterate, creator.Individual,toolbox.Indices)
        toolbox.register("getone", tools.initIterate, creator.Individual,toolbox.Getone)
        # ind1 = toolbox.Individual()

        toolbox.register("Population", tools.initRepeat, list, toolbox.Individual)


        toolbox.register("Evaluate", self.evaluate_fitness)

        # p = toolbox.Population(n=self.N)
        # fitnesses = map(toolbox.Evaluate, p)

        # for i in p:
        #     print(i)  # 打印每个个体的实数编码结果

        # for fit in fitnesses:
        #     print(fit)  # 打印每个个体的 fitness 值

        # 将个体 population 与对应的 fitness 绑定
        # for ind, fit in zip(p, fitnesses):
        #     ind.fitness.values = fit

            # print(ind.fitness.values)  # 打印每个个体的 fitness 值

        # 选择方式1：锦标赛选择
        toolbox.register('TourSel', tools.selTournament, tournsize=3)  # 注册 Tournsize 为 2 的锦标赛选择
        toolbox.register("select", tools.selNSGA2)
        # selectedTour = toolbox.TourSel(p, 3)  # 选择 5 个个体
        # print('锦标赛选择结果：')
        # for ind in selectedTour:
        #     print(ind)
        #     print(ind.fitness.values)


        # sys.exit()
        # toolbox.register("mate", self.crossover_operation_deap)
        # toolbox.register("mutate", self.mutation_operation_deap)
        # toolbox.register("select", tools.selNSGA2)

        return toolbox


    def initialize_population(self):
        population = []


        for _ in range(self.N):
            selected_indices_set = []
            individual = self.generate_individual()
            # selected_indices_set.update(selected_indices)  # 将生成个体的下标加入已选择集合
            population.append(individual)

        return np.array(population)

    def generate_individual(self):

        selected_indices_set=[]
        # 生成一个个体，并传入已经选过的下标集合
        R = np.random.randint(0, self.D+1)  # 选择的特征数量
        # print(R)
        selected_indices = self.binary_tournament(R, selected_indices_set)
        individual = np.zeros(self.D,dtype=int)
        individual[selected_indices] = 1

        return individual

    def binary_tournament(self, R, selected_indices_set):
        # 二进制竞赛选择R个特征的下标，排除之前选择过的下标
        selected_indices = []
        for _ in range(R):
            candidate1 = self.select_candidate(selected_indices_set)
            candidate2 = self.select_candidate(selected_indices_set)

            winner = self.select_winner(candidate1, candidate2)
            selected_indices.append(winner)
            selected_indices_set.append(winner)  # 将选出的下标加入已选择集合
        return selected_indices

    def select_candidate(self, selected_indices_set):
        # 选择候选下标，排除已经选过的下标
        candidate = np.random.randint(0, self.D)
        while candidate in selected_indices_set:
            candidate = np.random.randint(0, self.D)
        return candidate

    def select_winner(self, candidate1, candidate2):
        # 根据特征权重进行二进制竞赛选择胜者
        if self.sc[candidate1] > self.sc[candidate2]:
            return candidate1
        else:
            return candidate2

    # 其他方法保持不变
    def evaluate_fitness(self, individual):
        # 计算适应度，包括平均错误率和特征个数
        feature_indices = np.where(np.array(individual) == 1)[0]
        if (len(feature_indices)==0):
            return 1,len(individual)
        else:
            X_selected = self.x * individual
            # print(X_selected)
            # 适应度函数1：KNN分类器下进行K折交叉验证的平均错误率
            knn_classifier = KNeighborsClassifier(n_neighbors=5)
            cv = StratifiedKFold(n_splits=self.k_fold, shuffle=True, random_state=42)
            error_rates = 1 - cross_val_score(knn_classifier, X_selected, self.y, cv=cv, scoring='accuracy')

            # print(error_rates)
            avg_error_rate = np.mean(error_rates)
            # print(avg_error_rate)
            # 适应度函数2：特征个数
            # num_selected_features = len(feature_indices)
            return avg_error_rate, sum(individual)

        # print(individual)
        # print(len(feature_indices))
        # print(self.x)


    def intersection_two(self,L1,L2):
        return [1 if int(x) == int(y) == 1 else 0 for x, y in zip(L1, L2)]
    def crossover_operation(self,p1, p2, p3, sc):
        del p1.fitness.values
        del p2.fitness.values
        del p3.fitness.values

        L1 = self.intersection_two(p1, p2)
        L2 = self.intersection_two(p1, p3)
        L3 = self.intersection_two(p2, p3)
        # print(p1)
        # print(p2)
        # print(p3)
        #
        # print(L2)
        # print(L1)
        # print(L3)

        Oi = list(np.logical_or(np.logical_or(L1, L2).astype(int), L3).astype(int)) # selected more than twice
        S3 = self.intersection_two(self.intersection_two(L1, L2), L3)  # selected Three times
        S2 = list(np.array(Oi)-np.array(S3))
        p1_p2_p3=np.array(np.logical_or(np.logical_or(p1, p2).astype(int), p3).astype(int))
        s3_s2=np.array(np.logical_or(S3, S2).astype(int))
        S1 = list(p1_p2_p3-s3_s2)
        # print(Oi)
        # print('************')
        # print(S1)
        # print(S2)
        #
        # print(S3)


        O = np.copy(Oi)

        if np.random.rand() < 0.5:
            z1_idx, z2_idx = self.select_index(S2, 2)
            z_idx = self.compare_and_choose_larger(z1_idx, z2_idx, sc)
            O[z_idx] = 0
        else:
            o1_idx, o2_idx = self.select_index(S1, 2)
            o_idx = self.compare_and_choose_smaller(o1_idx, o2_idx, sc)
            O[o_idx] = 1

        return O

    def select_index(self,indices, num):
        selected_indices = np.random.choice(indices, size=num, replace=False)
        return selected_indices

    def compare_and_choose_larger(self,idx1, idx2, sc):
        return idx1 if sc[idx1] > sc[idx2] else idx2

    def compare_and_choose_smaller(self,idx1, idx2, sc):
        return idx1 if sc[idx1] < sc[idx2] else idx2



    def mutation_operation(self,O, sc):
        # print(O)
        if np.random.rand() < 0.5:
            # print(111)
            s_idx = np.where(O == 1)[0]
            # print(s_idx)
            if len(s_idx) == 0:
                return O
            z_idx = self.tournament_selection_smaller(s_idx, sc)
            # print(z_idx)
            O[z_idx] = 0
        else:
            # print(222)
            us_idx = np.where(O == 0)[0]
            # print(us_idx)
            if len(us_idx) == 0:
                return O
            o_idx = self.tournament_selection_larger(us_idx, sc)
            # print(o_idx)
            O[o_idx] = 1

        return O

    def tournament_selection_smaller(self,indices, sc):
        tournament_size = min(2, len(indices))
        selected_indices = np.random.choice(indices, size=tournament_size, replace=False)
        return min(selected_indices, key=lambda idx: sc[idx])

    def tournament_selection_larger(self,indices, sc):
        tournament_size = min(2, len(indices))
        selected_indices = np.random.choice(indices, size=tournament_size, replace=False)
        return max(selected_indices, key=lambda idx: sc[idx])


    def get_subset(self):
        toolbox=self.toolbox
        population = toolbox.Population(n=self.N)
        fitnesses = map(toolbox.Evaluate, population)

        for individual, fitness in zip(population, fitnesses):

            individual.fitness.values = fitness

        pareto_front =list()
        p_new=list()
        for generation in range(self.maxFEs):
            # Evaluate population fitness values

            # Select next generation individuals
            population_copy = list(map(toolbox.clone, population))
            offspring=list()
            for i in range(self.N):
                selectedTour = toolbox.TourSel(population_copy, 3)  # 选择 3 个个体
                selectedTour=list(map(toolbox.clone, selectedTour))
                oi=self.mutation_operation(self.crossover_operation(selectedTour[0],selectedTour[1],selectedTour[2],self.sc),self.sc)
                oi=list(oi)
                self.oi=oi
                oi=toolbox.getone()
                # print(oi.fitness.values)
                # sys.exit()
                offspring.append(oi)

            # Evaluate offspring
            fitnesses_off = list(map(toolbox.Evaluate, offspring))
            for ind, fit in zip(offspring, fitnesses_off):
                ind.fitness.values = fit

            p_new=p_new+offspring

            R=p_new+population

            # Select the next generation
            population = toolbox.select(R, len(population))


            # Pareto front of the final population
            pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
            print(f'第{generation}次结果：{pareto_front}')

        return pareto_front



    def get_oi(self):

        return self.oi

    def returnPAnd(self,list1,list2):
        # 合并两个列表
        merged_list = list1 + list2

        # 使用集合去除重复元素
        result_set = {tuple(item) for item in merged_list}

        # 转换回列表
        result = [list(item) for item in result_set]

        return result

if __name__ == '__main__':
    a=0
    # x,y=utiles.get_scaler_data('../data/final/xor_dataset.data')
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # a=NSGA2(100,300,10,[0.5,0.5,0.5,0.0,0,0,0,0,0,0],X_train,y_train)
    # a.get_subset()




