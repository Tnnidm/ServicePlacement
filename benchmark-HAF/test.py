import numpy as np
# import PDDPG_2D

a = np.array([[]])
print(a)

dict_a = {}
dict_a.update({1:None})
print(dict_a)
for i in dict_a.keys():
    if dict_a[i]==None:
        print(i)
b = np.zeros((2,2))
print(b)
c = np.concatenate((a, b), axis=0)
print(c)
