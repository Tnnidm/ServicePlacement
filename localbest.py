# Libs used
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import xlwt
import math
import csv


class SystemPerformance:
    def __init__(self, a_dim, ):
        self.a_dim = a_dim
        # self.spf = None
        self.wb = xlwt.Workbook(encoding="utf-8")
        self.ws = None


    def reset(self, Date, i_episode, State):
        if State == 0:
            self.ws = self.wb.add_sheet('Train_'+str(Date)+'_'+str(i_episode), cell_overwrite_ok=True)
        elif State == 1:
            self.ws = self.wb.add_sheet('Test_'+str(Date), cell_overwrite_ok=True)
        title = ['Time', 'Car Num', 'Open INDE Num', 'Reward', 'Car Connect Rate', 'Mean Workload', 'Mean Latency', 'Latency Outage Rate', 'AdaptSpeed']
        for i in range(len(title)):
            self.ws.write(0, i, title[i])

    def update(self, car_num, Open_INDE_Num, r, disconnect_rate, Avg_Delay, Delay_Outage_Rate, t):
        connect_rate = 1-disconnect_rate
        ave_open_uti_rate = 0
        self.ws.write(t+1, 0, t)
        self.ws.write(t+1, 1, int(car_num))
        self.ws.write(t+1, 2, int(Open_INDE_Num))
        self.ws.write(t+1, 3, int(r))
        self.ws.write(t+1, 4, float(connect_rate))
        self.ws.write(t+1, 5, float(ave_open_uti_rate))
        self.ws.write(t+1, 6, float(Avg_Delay))
        self.ws.write(t+1, 7, float(Delay_Outage_Rate))
        # if t%4319 == 0:
        #     self.store_xsl(path)


    def store_xsl(self):
        self.wb.save('SystemPerformance.xls')


a_dim = 1910
sys_per = SystemPerformance(a_dim)
for Date in range(1012,1032):
    print(Date)
    sys_per.reset(Date, 0, 0)
    t = 0
    with open('car_record/car_count_record_'+str(Date)+'.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            for i in range(len(row)):
                row[i] = float(row[i])
            arr = np.array(row[:len(row)-2])
            arr = arr.reshape(10,10)
            # print(arr)
            arr_sum = np.zeros((4,4))
            for i in range(3):
                for j in range(3):
                    arr_sum[i,j] = np.sum(arr[3*i:3*i+3, 3*j:3*j+3])
            for i in range(3):
                arr_sum[i,3] = np.sum(arr[3*i:3*i+3, 9])
            for j in range(3):
                arr_sum[3,j] = np.sum(arr[9, 3*j:3*j+3])
            arr_sum[3,3] = arr[9, 9]
            # print(arr_sum)
            open_INDE_sum = np.zeros((4,4))
            for i in range(4):
                for j in range(4):
                    open_INDE_sum[i,j] = math.ceil(arr_sum[i,j]/5)
            car_num = np.sum(arr_sum)
            INDE_num = np.sum(open_INDE_sum)
            reward = 6*car_num/5-3*INDE_num
            sys_per.update(car_num, INDE_num, reward, 0, 0, 0, t)
            t = t+1
    sys_per.store_xsl()