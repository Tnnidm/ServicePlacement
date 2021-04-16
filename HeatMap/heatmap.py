# Libs used
import numpy as np
from numpy import array as array
from matplotlib import pyplot as plt
import seaborn as sbn


f = open('../preprocessed_data/dict_Car/dict_Car_'+str(1012)+'.txt','r')  #读取每一天对应的字典
a = f.read()
dict_Car = eval(a)
f.close()

x_solution = 200
y_solution = 200
hp = np.zeros((4320,x_solution,y_solution))

for key in list(dict_Car.keys()):
    arr = dict_Car[key]
    # print(arr)
    if Ellipsis not in arr:
        for i in range(len(arr)):
            hp[int(arr[i,2]%4320),y_solution-1-int(y_solution*arr[i,1]/10000),int(x_solution*arr[i,0]/10000)] += 1

plt.figure(figsize = (1.8,1.4),dpi = 300)
sbn.set()

heatmap = sbn.heatmap(np.sum(hp[360:460, :, :], axis = 0), cbar = False, xticklabels = False, yticklabels = False)
plt.savefig('6.png', dpi = 1000, vmin=0, vmax=100)
plt.clf()

heatmap = sbn.heatmap(np.sum(hp[900:1000, :, :], axis = 0), cbar = False, xticklabels = False, yticklabels = False)
plt.savefig('9.png', dpi = 1000, vmin=0, vmax=100)
plt.clf()

heatmap = sbn.heatmap(np.sum(hp[1440:1540, :, :], axis = 0), cbar = False, xticklabels = False, yticklabels = False)
plt.savefig('12.png', dpi = 1000, vmin=0, vmax=100)
plt.clf()

heatmap = sbn.heatmap(np.sum(hp[1980:2080, :, :], axis = 0), cbar = False, xticklabels = False, yticklabels = False)
plt.savefig('15.png', dpi = 1000, vmin=0, vmax=100)
plt.clf()

heatmap = sbn.heatmap(np.sum(hp[2520:2620, :, :], axis = 0), cbar = False, xticklabels = False, yticklabels = False)
plt.savefig('18.png', dpi = 1000, vmin=0, vmax=100)
plt.clf()

heatmap = sbn.heatmap(np.sum(hp[3060:3160, :, :], axis = 0), cbar = False, xticklabels = False, yticklabels = False)
plt.savefig('21.png', dpi = 1000, vmin=0, vmax=100)
plt.clf()