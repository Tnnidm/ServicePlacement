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

for i in range(0,int(4320/9)):
    heatmap = sbn.heatmap(np.sum(hp[9*i:9*(i+1), 100:200, 50:150], axis = 0), xticklabels = False, yticklabels = False, vmin=0, vmax=10, cmap='RdYlGn_r')
    plt.savefig('heatmap/'+str(int(3*i/60)).zfill(2)+str(int((3*i)%60)).zfill(2)+".png", dpi = 1000)
    plt.clf()
