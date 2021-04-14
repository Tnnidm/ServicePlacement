# Libs used
import numpy as np
import random
import copy
import math
# Modules used
import config

SLOTNUM = config.SLOTNUM # the numbers of slots used
a_dim = config.a_dim
s_dim = config.s_dim



class Localbest(object):
    def __init__(self, a_dim):
        self.a_dim = a_dim
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
        self.a = np.zeros((1, self.a_dim)) 


    def choose_action(self, env, t):
        # env_pre = copy.deepcopy(env)
        # env_pre.update_only_car(t)
        if t%5 == 0:
            CarPos = env.GetCarPos()
            # if t in list(env.dict_Car_TimeIndex.keys()):
            #     for i in range(len(env.dict_Car_TimeIndex[t])):
            #         if Ellipsis not in env.dict_Car[env.dict_Car_TimeIndex[t][i]]:
            #             # print((env.dict_Car[env.dict_Car_TimeIndex[t][i]])[0][0:2])
            #             CarPos = np.concatenate((CarPos, (env.dict_Car[env.dict_Car_TimeIndex[t][i]])[0:1][0:2]), axis = 0)
                        
            self.a = np.zeros((1, self.a_dim))
            if len(CarPos) != 0:
                open_num = np.zeros(100,)
                for i in range(len(CarPos)):
                    cell_num = 10*int(CarPos[i][0]/1000)+int(CarPos[i][1]/1000)
                    open_num[cell_num] = open_num[cell_num]+1

                for i in range(100):
                    open_num[i] = math.ceil(open_num[i]/5)
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

