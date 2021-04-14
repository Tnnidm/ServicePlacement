# Libs used
import numpy as np
# import random
# from sklearn.cluster import KMeans
# import math
import copy
import gc
# Modules used
import config


SLOTNUM = config.SLOTNUM # the numbers of slots used
a_dim = config.a_dim
s_dim = config.s_dim



class HAF(object):
    def __init__(self, a_dim):
        self.a_dim = a_dim
        self.INDE_pos = np.loadtxt(config.preprocessed_data_Location+'bt_inside_1910.txt')
        self.INDE_open = np.ones((self.a_dim))
        self.INDE_coned_times = np.zeros((self.a_dim))

        self.INDE_con_car = {}
        for i in range(a_dim):
            self.INDE_con_car.update({i:[]})

        self.CarInfo = None
        # self.INDE_pos = np.loadtxt(config.preprocessed_data_Location+'bt_inside_1910.txt')
        self.dict_L1_INDE_cell = {}
        for i in range(100):
            self.dict_L1_INDE_cell.update({i:[]})
        for i in range(len(self.INDE_pos)):
            cell_num = 10*int(self.INDE_pos[i][0]/1000)+int(self.INDE_pos[i][1]/1000)
            self.dict_L1_INDE_cell[cell_num].append(i)
        # for i in range(100):
        #     random.shuffle(self.dict_L1_INDE_cell[i])
            # print(self.dict_L1_INDE_cell[i]) 
        self.dict_CarInfo_statistic = {}
        for i in range(4320):
            self.dict_CarInfo_statistic.update({i:np.zeros((0,0))})


    def collect_info(self, env, t):
        if len(self.dict_CarInfo_statistic[t]) == 0:
            self.dict_CarInfo_statistic[t] = env.GetCarInfo()
        else:
            self.dict_CarInfo_statistic[t] = np.concatenate((self.dict_CarInfo_statistic[t], env.GetCarInfo()), axis=0)

    def choose_action(self, env):
        action = np.zeros((1, self.a_dim))
        self.CarInfo = env.GetCarInfo()
        self.INDE_open = np.ones((self.a_dim))
        count = 0
        Capacity = 5
        self.INDE_coned_times = np.zeros((self.a_dim))

        self.INDE_con_car = {}
        for i in range(a_dim):
            self.INDE_con_car.update({i:[]})

        for i in range(len(self.CarInfo)):
            self.CarInfo[i,2] = -1

        # print('length of CarInfo = '+str(len(self.CarInfo)))

        while((self.CarInfo[:,2] == -2).all()==False):
            flag = 0            
            for i in range(len(self.CarInfo)):
                if self.CarInfo[i,2] == -1:
                    count = count+1
                    pos0 = self.CarInfo[i,0]
                    pos1 = self.CarInfo[i,1]
                    cell_num = 10*int(pos0/1000)+int(pos1/1000)
                    distance = 999999
                    for j in range(len(self.dict_L1_INDE_cell[cell_num])):
                        INDE_No = (self.dict_L1_INDE_cell[cell_num])[j]
                        if self.INDE_open[INDE_No] == 1:
                            INDEpos0 = self.INDE_pos[j][0]
                            INDEpos1 = self.INDE_pos[j][1]
                            newdistance = np.sqrt(np.square(INDEpos0-pos0)+np.square(INDEpos1-pos1))
                            if newdistance<distance:
                                distance = newdistance
                                self.CarInfo[i,2] = INDE_No
                    if self.CarInfo[i,2] != -1:
                        # if i >= len(self.CarInfo):
                            # print('error car')
                            # print(i)
                            # print(self.CarInfo[i,2])
                        self.INDE_con_car[int(self.CarInfo[i,2])].append(i)
                        self.INDE_coned_times[int(self.CarInfo[i,2])] += 1
                        flag = 1
            if flag == 0:
                break
            # print(self.INDE_con_car)
            max_INDE = int(np.argmax(self.INDE_coned_times))
            max_num = int(np.max(self.INDE_coned_times))
            if max_num>Capacity:
                for k in range(self.a_dim):
                    if self.INDE_coned_times[k]>=Capacity:
                        self.INDE_open[k] = 0
                        action[0, k] = 1
                        for i in range(len(self.INDE_con_car[k])):
                            if i < Capacity:
                                # print(k)
                                # print((self.INDE_con_car[k])[i])
                                self.CarInfo[(self.INDE_con_car[k])[i],2] = -2
                            else:
                                self.CarInfo[(self.INDE_con_car[k])[i],2] = -1
                        self.INDE_con_car[k] = []
                        self.INDE_coned_times[k] = 0
            else:
                for k in range(self.a_dim):
                    if self.INDE_coned_times[k]>0:
                        self.INDE_open[k] = 0
                        action[0, k] = 1
                        for i in range(len(self.INDE_con_car[k])):
                            if i < Capacity:
                                self.CarInfo[self.INDE_con_car[k][i],2] = -2
                            else:
                                self.CarInfo[self.INDE_con_car[k][i],2] = -1
                        self.INDE_con_car[k] = []
                        self.INDE_coned_times[k] = 0                    


        return action, count

    def choose_action_statistic(self, env, Date, t):
        action = np.zeros((1, self.a_dim))
        self.CarInfo = self.dict_CarInfo_statistic[t]
        self.INDE_open = np.ones((self.a_dim))
        count = 0
        Capacity = 5*(Date-1008)
        self.INDE_coned_times = np.zeros((self.a_dim))

        self.INDE_con_car = {}
        for i in range(a_dim):
            self.INDE_con_car.update({i:[]})
            
        for i in range(len(self.CarInfo)):
            self.CarInfo[i,2] = -1

        while((self.CarInfo[:,2] == -2).all()==False):
            flag = 0            
            for i in range(len(self.CarInfo)):
                if self.CarInfo[i,2] == -1:
                    count = count+1
                    pos0 = self.CarInfo[i,0]
                    pos1 = self.CarInfo[i,1]
                    cell_num = 10*int(pos0/1000)+int(pos1/1000)
                    distance = 999999
                    for j in range(len(self.dict_L1_INDE_cell[cell_num])):
                        INDE_No = (self.dict_L1_INDE_cell[cell_num])[j]
                        if self.INDE_open[INDE_No] == 1:
                            INDEpos0 = self.INDE_pos[j][0]
                            INDEpos1 = self.INDE_pos[j][1]
                            newdistance = np.sqrt(np.square(INDEpos0-pos0)+np.square(INDEpos1-pos1))
                            if newdistance<distance:
                                distance = newdistance
                                self.CarInfo[i,2] = INDE_No
                    if self.CarInfo[i,2] != -1:
                        self.INDE_con_car[int(self.CarInfo[i,2])].append(i)
                        self.INDE_coned_times[int(self.CarInfo[i,2])] += 1
                        flag = 1
            if flag == 0:
                break

            max_INDE = int(np.argmax(self.INDE_coned_times))
            max_num = int(np.max(self.INDE_coned_times))
            if max_num>Capacity:
                for k in range(self.a_dim):
                    if self.INDE_coned_times[k]>Capacity:
                        self.INDE_open[k] = 0
                        action[0, k] = 1
                        for i in range(len((self.INDE_con_car[k]))):
                            if i < Capacity:
                                self.CarInfo[self.INDE_con_car[k][i],2] = -2
                            else:
                                self.CarInfo[self.INDE_con_car[k][i],2] = -1
                        self.INDE_con_car[k] = []
                        self.INDE_coned_times[k] = 0
            else:
                for k in range(self.a_dim):
                    if self.INDE_coned_times[k]>0:
                        self.INDE_open[k] = 0
                        action[0, k] = 1
                        for i in range(len(self.INDE_con_car[k])):
                                self.CarInfo[self.INDE_con_car[k][i],2] = -2
                        self.INDE_con_car[k] = []
                        self.INDE_coned_times[k] = 0

        return action, count
    # def choose_action(self, i_p, t, a_last, env):
    #     action = np.zeros((1, self.a_dim))

    #     a_pre = np.ones((self.a_dim))
    #     a_last_pre = a_last
    #     env_pre = copy.deepcopy(env)
    #     index = 0
    #     count = 0
    #     while(1):
    #         index, num = env_pre.update_imaginarily(a_pre, a_last_pre, i_p, t)
    #         count = count+num
    #         if index != -1:
    #             a_last_pre = a_pre
    #             a_pre[index] = 0
    #             action[0,index] = 1
    #         else:
    #             break
    #     del env_pre
    #     gc.collect()

    #     return action, count

