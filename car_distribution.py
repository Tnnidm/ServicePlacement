import numpy as np
import csv
from numpy import array as array
from datetime import datetime
import os, shutil


dt=datetime.now() #创建一个datetime类对象
DAY =  str(dt.month).zfill(2)+str(dt.day).zfill(2)
Time = str(dt.month).zfill(2)+str(dt.day).zfill(2)+'_'+str(dt.hour).zfill(2)+str(dt.minute).zfill(2)
del dt
path = 'result/'+DAY
if os.path.exists(path) == False:
    os.makedirs(path);
path = path+'/'+Time
if os.path.exists(path) == False:
    os.makedirs(path);


for Date in range(1001,1032):
    f = open('preprocessed_data/dict_Car/dict_Car_'+str(Date)+'.txt','r')
    a = f.read()
    dict_Car = eval(a)
    f.close()

    f = open('preprocessed_data/dict_Car/dict_Car_'+str(Date)+'_TimeIndex.txt','r')
    a = f.read()
    dict_Car_TimeIndex = eval(a)
    f.close()        
    del a

    car_count = np.zeros((4320,102))

    for t in range(0+(Date-1001)*4320, 4320+(Date-1001)*4320):
        if t in list(dict_Car_TimeIndex.keys()):
            for i in range(len(dict_Car_TimeIndex[t])):
                if Ellipsis not in dict_Car[dict_Car_TimeIndex[t][i]]:
                    place = dict_Car[dict_Car_TimeIndex[t][i]]
                    for j in range(len(place)):
                        t_p = int(place[j][2])-(Date-1001)*4320
                        if t_p in range(4320):
                            cell_number = int(10*int(place[j][0]/1000)+int(place[j][1]/1000))
                            car_count[t_p,cell_number] += 1

    for t in range(4320):
        car_count[t,101] = np.sum(car_count[t,0:100])
        for i in range(100):
            if car_count[t,i]>0:
                car_count[t,100] += 1

    car_count_record = open(path+'/car_count_record_'+str(Date)+'.csv','w',encoding='utf-8')
    csv_writer = csv.writer(car_count_record)
    for i in range(4320):
        csv_writer.writerow(car_count[i].tolist())
    car_count_record.close()