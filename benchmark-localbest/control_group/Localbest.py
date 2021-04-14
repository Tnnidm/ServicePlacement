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
        # for i in range(100):
        #     random.shuffle(self.dict_L1_INDE_cell[i])
        #     # print(self.dict_L1_INDE_cell[i]) 
        self.a = np.zeros((1, self.a_dim)) 


    def choose_action(self, env, t):
        # env_pre = copy.deepcopy(env)
        # env_pre.update_only_car(t)

            # if t in list(env.dict_Car_TimeIndex.keys()):
            #     for i in range(len(env.dict_Car_TimeIndex[t])):
            #         if Ellipsis not in env.dict_Car[env.dict_Car_TimeIndex[t][i]]:
            #             # print((env.dict_Car[env.dict_Car_TimeIndex[t][i]])[0][0:2])
            #             CarPos = np.concatenate((CarPos, (env.dict_Car[env.dict_Car_TimeIndex[t][i]])[0:1][0:2]), axis = 0)
                        
        self.a = np.zeros((1, self.a_dim))
        for cell_num in range(100):
            self.a[0, int(self.dict_L1_INDE_cell[cell_num][0])] = 1                


        return self.a

