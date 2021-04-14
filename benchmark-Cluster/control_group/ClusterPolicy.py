# Libs used
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import math
# Modules used
import config


SLOTNUM = config.SLOTNUM # the numbers of slots used
a_dim = config.a_dim
s_dim = config.s_dim



class Cluster(object):
    def __init__(self, a_dim):
        self.a_dim = a_dim
        self.a = np.zeros((1, self.a_dim))
        self.INDE_pos = np.loadtxt(config.preprocessed_data_Location+'bt_inside_1910.txt')
        self.dict_L1_INDE_cell = {}
        for i in range(100):
            self.dict_L1_INDE_cell.update({i:[]})
        for i in range(len(self.INDE_pos)):
            cell_num = 10*int(self.INDE_pos[i][0]/1000)+int(self.INDE_pos[i][1]/1000)
            self.dict_L1_INDE_cell[cell_num].append(i)
        for i in range(100):
            random.shuffle(self.dict_L1_INDE_cell[i])
            # print(self.dict_L1_INDE_cell[i]) 
        self.dict_CarPos_statistic = {}
        for i in range(4320):
            self.dict_CarPos_statistic.update({i:np.zeros((0,0))})


    # def collect_info(self, env, t):
    #     if len(self.dict_CarPos_statistic[t]) == 0:
    #         self.dict_CarPos_statistic[t] = env.GetCarPos()
    #     else:
    #         self.dict_CarPos_statistic[t] = np.concatenate((self.dict_CarPos_statistic[t], env.GetCarPos()), axis=0)
    def collect_info(self, env, t):
        if len(self.dict_CarPos_statistic[t]) == 0:
            self.dict_CarPos_statistic[t] = env.GetCarPos()
        else:
            self.dict_CarPos_statistic[t] = np.concatenate((self.dict_CarPos_statistic[t], env.GetCarPos()), axis=0)

    def choose_action(self, env):
        CarPos = env.GetCarPos()
        self.a = np.zeros((1, self.a_dim))
        if len(CarPos) != 0:
            K = math.ceil(len(CarPos)/5)
            # estimator = KMeans(n_clusters=K, n_jobs=6)
            # estimator.fit(CarPos)
            # label_pred = (estimator.labels_).tolist()
            # centroids = estimator.cluster_centers_

            estimator = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=1000)
            estimator.fit(CarPos)
            K = estimator.n_clusters_
            label_pred = estimator.labels_
            centroids = np.zeros((K,2))
            for i in range(K):            
                place = np.where(label_pred == i)
                place = place[0].tolist()
                # print(place)
                Pos = CarPos[place,:]
                centroids[i] = np.mean(Pos, axis = 0)
            label_pred = label_pred.tolist()

            open_num = np.zeros(100,)
            for i in range(K):
                car_num = label_pred.count(i)     
                cell_num = 10*int(centroids[i][0]/1000)+int(centroids[i][1]/1000)
                open_num[cell_num] = open_num[cell_num]+car_num/5
            for i in range(100):
                open_num[i] = math.ceil(open_num[i])
                # print(self.dict_L1_INDE_cell[i])
                # print(open_num[i])
                if open_num[i]<=len(self.dict_L1_INDE_cell[i]):
                    open_list = self.dict_L1_INDE_cell[i][0:int(open_num[i])]
                else:
                    open_list = self.dict_L1_INDE_cell[i]
                # print(open_list)
                for j in range(len(open_list)):
                    self.a[0, int(open_list[j])] = 1

        return self.a

    def choose_action_t(self, t):
        CarPos = self.dict_CarPos_statistic[t]
        self.a = np.zeros((1, self.a_dim))
        if len(CarPos) != 0:
            K = math.ceil(len(CarPos)/10)
            # estimator = KMeans(n_clusters=K, n_jobs=6)
            # estimator.fit(CarPos)
            # label_pred = (estimator.labels_).tolist()
            # centroids = estimator.cluster_centers_

            estimator = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=1000)
            estimator.fit(CarPos)
            K = estimator.n_clusters_
            label_pred = estimator.labels_
            centroids = np.zeros((K,2))
            for i in range(K):            
                place = np.where(label_pred == i)
                place = place[0].tolist()
                # print(place)
                Pos = CarPos[place,:]
                centroids[i] = np.mean(Pos, axis = 0)
            label_pred = label_pred.tolist()

            open_num = np.zeros(100,)
            for i in range(K):
                car_num = label_pred.count(i)     
                cell_num = 10*int(centroids[i][0]/1000)+int(centroids[i][1]/1000)
                open_num[cell_num] = open_num[cell_num]+car_num/5
            for i in range(100):
                open_num[i] = math.ceil(open_num[i])
                # print(self.dict_L1_INDE_cell[i])
                # print(open_num[i])
                if open_num[i]<=len(self.dict_L1_INDE_cell[i]):
                    open_list = self.dict_L1_INDE_cell[i][0:int(open_num[i])]
                else:
                    open_list = self.dict_L1_INDE_cell[i]
                # print(open_list)
                for j in range(len(open_list)):
                    self.a[0, int(open_list[j])] = 1

        return self.a

    def choose_action_statistic(self, Date, t):
        CarPos = self.dict_CarPos_statistic[t]
        self.a = np.zeros((1, self.a_dim))
        if len(CarPos) != 0:
            K = math.ceil(len(CarPos)/(5*(Date-1008)))
            # estimator = KMeans(n_clusters=K, n_jobs=6)
            # estimator.fit(CarPos)
            # label_pred = (estimator.labels_).tolist()
            # centroids = estimator.cluster_centers_

            estimator = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=1000)
            estimator.fit(CarPos)
            K = estimator.n_clusters_
            label_pred = estimator.labels_
            centroids = np.zeros((K,2))
            for i in range(K):            
                place = np.where(label_pred == i)
                place = place[0].tolist()
                # print(place)
                Pos = CarPos[place,:]
                centroids[i] = np.mean(Pos, axis = 0)
            label_pred = label_pred.tolist()

            open_num = np.zeros(100,)
            for i in range(K):
                car_num = label_pred.count(i)     
                cell_num = 10*int(centroids[i][0]/1000)+int(centroids[i][1]/1000)
                open_num[cell_num] = open_num[cell_num]+car_num/5
            for i in range(100):
                open_num[i] = math.ceil(open_num[i])
                # print(self.dict_L1_INDE_cell[i])
                # print(open_num[i])
                if open_num[i]<=len(self.dict_L1_INDE_cell[i]):
                    open_list = self.dict_L1_INDE_cell[i][0:int(open_num[i])]
                else:
                    open_list = self.dict_L1_INDE_cell[i]
                # print(open_list)
                for j in range(len(open_list)):
                    self.a[0, int(open_list[j])] = 1

        return self.a


