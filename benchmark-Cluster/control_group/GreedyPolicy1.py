# Libs used
import numpy as np
import random
# Modules used
import config

SLOTNUM = config.SLOTNUM # the numbers of slots used
a_dim = config.a_dim
s_dim = config.s_dim



MemoryLength = 5

class Greedy(object):
    def __init__(self, a_dim):
        self.a_dim = a_dim

        self.a_mem = np.zeros((MemoryLength,a_dim))
        self.s_mem = np.zeros((MemoryLength,a_dim))

    def choose_action(self, env, s, a_last):
        a = np.zeros((1, self.a_dim))
        if np.sum(a_last) == 0:
            for i in range(self.a_dim):
                if random.random()< 0.2:
                    a[0,i] = random.uniform(0.6, 0.9)
                else:
                    a[0,i] = random.uniform(0.1, 0.4)
        else:
            for i in range(self.a_dim):
                if s[0,i] != 0:
                    a[0,i] = random.uniform(0.6, 0.9)
                    if s[0,i] >= 1:
                        neighbor = (env.list_INDE_object[i]).neighbor
                        for j in range(len(neighbor)):
                            closest_neighbor_id = int(neighbor[j,0])
                            if a_last[closest_neighbor_id] == 0:
                                if random.random()< 0.6:
                                    a[0,closest_neighbor_id] = random.uniform(0.6, 0.9)
                                break
                else:
                    if a_last[i] == 0:
                        # a[0,i] = random.uniform(0.1, 0.4)
                        pass
                    else:
                        if np.sum(self.a_mem[:,i])==MemoryLength and np.sum(self.s_mem[:,i])==0:
                            a[0,i] = random.uniform(0.1, 0.4)
                        else:
                            a[0,i] = random.uniform(0.6, 0.9)

        self.a_mem[1:MemoryLength,:] = self.a_mem[0:MemoryLength-1,:]
        self.a_mem[0] = a_last
        self.s_mem[1:MemoryLength,:] = self.s_mem[0:MemoryLength-1,:]
        self.s_mem[0,:] = s

        return a

# g = Greedy(3)
# env = 1

# # print(s)
# a_last = np.ones(3)
# for t in range(5):
#     s = np.random.rand(1,3)
#     a = g.choose_action(env, s, a_last)
#     a_last = a
#     print(s)
#     print(g.s_mem)
#     print(g.a_mem)
