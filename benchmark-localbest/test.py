import numpy as np
# import PDDPG_2D

a = np.ones((10,3))
print(a)
a[8,0] = 0

print(np.where(a == 0))
print(int(np.where(a == 0)[0]))

print((a[:,2] == 1).any()==False)