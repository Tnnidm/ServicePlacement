# Libs used
import numpy as np
from numpy import array as array
from matplotlib import pyplot as plt

f = open('dict_Car_1012.txt','r')  #读取每一天对应的字典
a = f.read()
dict_Car = eval(a)
f.close()
print("Open source successflly!")

temporal_map = {}
for t in range(4320):
    temporal_map.update({t:[]})

for key in list(dict_Car.keys()):
    arr = dict_Car[key]
    # print(arr)
    if Ellipsis not in arr:
        for i in range(len(arr)):
            time = int(arr[i,2]%4320)
            temporal_map[time].append([arr[i, 0], arr[i, 1]])
            # hp[,y_solution-1-int(y_solution*arr[i,1]/10000),int(x_solution*arr[i,0]/10000)] += 1
print("Finish temporal map successflly!")

# plt.figure(figsize = (9, 7))
plt.figure(figsize = (10, 9))


for i in range(0,int(4320/3)):
    temporal_map_sum = []
    for t in range(3*i, 3*(i+1)):
        temporal_map_sum = temporal_map_sum + temporal_map[t]
    temporal_map_sum = np.array(temporal_map_sum)
    plt.ylim(0, 5000)
    plt.xlim(2500, 7500)
    plt.scatter(temporal_map_sum[:,0], temporal_map_sum[:,1], c = 'b', marker = '.')
    plt.savefig('grid/'+str(int(i/60)).zfill(2)+str(int((i)%60)).zfill(2)+".png")
    
    plt.clf()
# np.save("846", np.sum(hp[846:855, 50:100, 25:75], axis = 0))


# ========================= Test Part ====================================
# arr = np.load("846.npy")
# plt.figure(figsize = (1.8,1.4),dpi = 300)
# sbn.set()
# new_RdYlGn_r=sbn.color_palette("RdYlGn_r", 20)[0:20]
# heatmap = sbn.heatmap(arr, xticklabels = False, yticklabels = False, vmin=0, vmax=10, cmap="RdYlGn_r")
# plt.savefig("1.png", dpi = 1000)
# plt.clf()
 # =======================================================================

# plt.figure(figsize = (1.8,1.4),dpi = 300)
# sbn.set()

# # new_RdYlGn_r=sns.color_palette("RdYlGn_r", 20)[5:20]

# for i in range(0,int(4320/9)):
# # for i in range(90,110):
#     heatmap = sbn.heatmap(np.sum(hp[9*i:9*(i+1), int(x_solution/2):x_solution, int(y_solution/4):int(3*y_solution/4)], axis = 0),\
#      xticklabels = False, yticklabels = False, vmin=0, vmax=15, cmap='RdYlGn_r')
#     plt.savefig('grid/'+str(int(3*i/60)).zfill(2)+str(int((3*i)%60)).zfill(2)+".png", dpi = 1000)
#     plt.clf()
