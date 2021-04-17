# Libs used
import numpy as np
from matplotlib import pyplot as plt
import torch
from datetime import datetime
import os, shutil
import xlwt
import copy
import gc
import csv
import random
import argparse
import wandb

# Modules used
import envirment_PDDPG
import output_PDDPG_2D
import PDDPG_2D
import control_group.GreedyPolicy
import config


def add_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=int, default=6,
                        help='# the reward coefficient of utilizing')

    parser.add_argument('--beta', type=int, default=3,
                        help='# the cost coefficient of opening a BS')

    parser.add_argument('--omega', type=int, default=5,
                        help='# the penalty coefficient of disconnection')

    parser.add_argument('--bs_capacity_method', type=str, default='same',
                        help='# are capacities of BSs same or different?')

    parser.add_argument('--notes', type=str, default='default',
                        help='# experiment notes')

    args = parser.parse_args()
    return args

args = add_args()



# some parameters
MAX_EPISODES = config.MAX_EPISODES
SLOTNUM = config.SLOTNUM # the numbers of slots used
Bt_INDE_num = config.Bt_INDE_num

# about utilization ratio punishment
U_BEST = config.U_BEST # as the name say
PHI_1 = config.PHI_1 # means the punishment of utilization ratio when u<u_best
PHI_2 = config.PHI_2 #means the punishment of utilization ratio when u>u_best

# about INDE cost
ALPHA = config.ALPHA # means the cost of maintaining a basic station INDE
BETA = config.BETA # means the cost of maintaining a car INDE

# about centent loss punishment
OMEGA = config.OMEGA # means the punishment of centent loss

# Hyper Parameters
MEMORY_CAPACITY = config.MEMORY_CAPACITY

a_dim = config.a_dim
s_dim = config.s_dim
use_gpu = config.use_gpu

EPSILON = config.EPSILON
var = 0.2




def GetPyFile(filePath):
    result = []
    for filename in os.listdir(filePath):
        fileinfo=os.path.splitext(filename)
        if fileinfo[1]=='.py':
            result.append(filename)
    return result

print('Initializing...')

print('use_gpu = '+str(use_gpu))

dt=datetime.now() #创建一个datetime类对象
DAY =  str(dt.month).zfill(2)+str(dt.day).zfill(2)
Time = str(dt.month).zfill(2)+str(dt.day).zfill(2)+'_'+str(dt.hour).zfill(2)+str(dt.minute).zfill(2)
del dt
path = 'result/'+DAY
if os.path.exists(path) == False:
    os.makedirs(path);
path = path+'/'+Time
random_str = ''
if os.path.exists(path) == True:
    random_str = '_'+str(random.randint(1,999999))
    path = path+random_str

os.makedirs(path);
os.makedirs(path+'/car_L1_reward_figure')
os.makedirs(path+'/AdaptSpeed')
os.makedirs(path+'/ConvergenceRate')
os.makedirs(path+'/log')
os.makedirs(path+'/result_images')
os.makedirs(path+'/SystemPerformance')
os.makedirs(path+'/code')

sourcefolder = os.getcwd()
desfolder = sourcefolder+'/'+path+'/code'
filelist=GetPyFile(sourcefolder+'/7')
for file in filelist:
    old_pos = sourcefolder+'/7/'+file
    new_pos = desfolder+'/'+file
    shutil.copyfile(old_pos, new_pos)
shutil.copyfile(sourcefolder+'/7/control_group/GreedyPolicy.py', desfolder+'/GreedyPolicy.py')

para_record = open(path+'/Para.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(para_record)


# workbook = xlwt.Workbook(encoding="utf-8")
# worksheet_h = workbook.add_sheet('train_times-loss_h', cell_overwrite_ok=True)
# worksheet_w = workbook.add_sheet('train_times-loss_w', cell_overwrite_ok=True)
# title = ['Time', 'Train Times', 'loss_a', 'td_error']
# for i in range(len(title)):
#     worksheet_h.write(0, i, title[i])
#     worksheet_w.write(0, i, title[i])
# work_line_h = 0
# work_line_w = 0
# loss_h = open(path+'/loss_h.csv', 'w', encoding='utf-8', newline='')
# csv_loss_h = csv.writer(loss_h)
# csv_loss_h.writerow(['Time', 'Train Times', 'loss_a', 'td_error'])

loss_w = open(path+'/loss_w.csv', 'w', encoding='utf-8', newline='')
csv_loss_w = csv.writer(loss_w)
csv_loss_w.writerow(['Time', 'Train Times', 'loss_a', 'td_error'])

'''
main program
'''
logger.info("--------global parameters setting-------")
global_hyp = dict()
for k, v in config.__dict__.items():
    if type(v) in [int, float, str, bool] and not k.startswith('_'):
        global_hyp.update({k: v})
for k, v in args.__dict__.items():
    if type(v) in [int, float, str, bool] and not k.startswith('_'):
        global_hyp.update({k: v})
# initialize the wandb.
if args.bs_capacity_method == "different":
    group_name = args.model+"-G2"
    experiment_name = "DeepReserve" + "-capacity" + str(args.bs_capacity_method) + "-" + Time + random_str
else:
    group_name = args.model+"-G1"
    experiment_name = "DeepReserve" + "-alpha" + str(args.alpha) + "-beta" + str(args.beta) + "-omega" + str(args.omega)
    
wandb.init(
    project=group_name,
    name=experiment_name,
    config=global_hyp
)


# dqn = dqn.DQN()
Agent = control_group.GreedyPolicy.Greedy(a_dim)

# ddpg_h = PDDPG_2D.PDDPG(a_dim, s_dim)
ddpg_w = PDDPG_2D.PDDPG(a_dim, s_dim)
# ddpg = ddpg.DDPG(a_dim, s_dim)
# retarder = Queue()
sys_per = output_PDDPG_2D.SystemPerformance(a_dim)
line_fig = output_PDDPG_2D.LineFigures(a_dim)
# conv_rate = output_PDDPG_2D.ConvergenceRate(MAX_EPISODES, SLOTNUM, a_dim, s_dim)
# adapt_speed = output_PDDPG_2D.AdaptSpeed(SLOTNUM, a_dim)




state = np.zeros((1, 10, 1, 1, 1910))
state_ = np.zeros((1, 10, 1, 1, 1910))


for Date in range(1008, 1012):
    print('Date = '+str(Date))

    env = envirment_PDDPG.Env(Date, path, args)
    a = np.zeros((1,a_dim))
    for i in range(a_dim):
        if np.random.uniform()<0.2:
            a[0,i] = 1
    a = np.ravel(a)

    a_last = np.ravel(np.zeros((1,a_dim)))
    s = np.ravel(np.zeros((1,s_dim)))
    s_ = np.ravel(np.zeros((1,s_dim)))

    loadrate, mixed_loadrate = env.update(a, a_last, 0, 0)
    s = mixed_loadrate

    s_ = s
    a_last = a
    s = np.transpose(s)
    state_[0,0:9,0] = state_[0,1:10,0]
    state[0,9,0] = s
    # print(state)
    s_ = np.transpose(s_)

    i_episode = 0

    sys_per.reset(Date, i_episode, 0)
    line_fig.reset()
    # adapt_speed.reset(SLOTNUM, a_dim)
    # conv_rate.reset(Date, path)
    # if Date == 1013:
    #     for target_param, param in zip(ddpg_h.Critic_target.parameters(), ddpg_w.Critic_target.parameters()):
    #         target_param.data.copy_(param.data)
    #     for target_param, param in zip(ddpg_h.Actor_target.parameters(), ddpg_w.Actor_target.parameters()):
    #         target_param.data.copy_(param.data)
    #     for target_param, param in zip(ddpg_h.Critic_eval.parameters(), ddpg_w.Critic_eval.parameters()):
    #         target_param.data.copy_(param.data)
    #     for target_param, param in zip(ddpg_h.Actor_eval.parameters(), ddpg_w.Actor_eval.parameters()):
    #         target_param.data.copy_(param.data)

    for t in range(1, SLOTNUM):
        # if t%10 == 0:
        #     print(t)
        # csv_writer.writerow([i_episode, t])
        # csv_writer.writerow(s[0].tolist())


        # if i_episode == 0:
        # if Date == 1008:
        #     # print(s)
        #     aa = Agent.choose_action(env, s, a_last)
        #     a_t = copy.deepcopy(aa)
        #     a = aa[0,:]                     
        # else:
        #     if Date<=1007 or (Date-1007)%7>=6:
        #         aa = ddpg_h.choose_action_2(state, EPSILON)
        #     else:
        #         aa = ddpg_w.choose_action_2(state, EPSILON)
        #     a_t = copy.deepcopy(aa).numpy()
        #     a = aa[0,:1910].numpy() 

        if Date == 1008:
            # print(s)
            aa = Agent.choose_action(env, s, a_last)
            a_t = copy.deepcopy(aa)
            a = aa[0,:]                     
        else:
            aa = ddpg_w.choose_action_2(state, env, a_last, EPSILON)
            a_t = copy.deepcopy(aa).numpy()
            a = aa[0,:1910].numpy() 


        for i in range(a_dim):
            if a[i] >= 0.5:
                a[i] = 1
            else:
                a[i] = 0

        # csv_writer.writerow(a.tolist())

        
        # open_state = env.Report_open_close()
        # for i in range(a_dim):
        #     if a_last[i]!=open_state[i]:
        #         print('1_ERROR!!!!!')
        #         break

        # if i_episode%3==0 and t%100==0:
        #     result_list = adapt_speed.Calculate_AdaptSpeed(env, a_last, i_episode, t, Date, SLOTNUM)
        #     reward_list = result_list[0] 
        #     disconnect_rate_list = result_list[1]
        #     Delay_Outage_Rate_list = result_list[2]

        loadrate, mixed_loadrate = env.update(a, a_last, i_episode, t+(Date-1001)*SLOTNUM)
        s_ = mixed_loadrate
        disconnect_rate, Avg_Delay, Delay_Outage_Rate = output_PDDPG_2D.Calculate_disconnect_outofdelay(env)
        r = output_PDDPG_2D.Calculate_Reward_new(a, loadrate, disconnect_rate, Delay_Outage_Rate, env, args)


        # if i_episode%3==0 and t%100==0:
        #     reward_list.append(float(r))
        #     disconnect_rate_list.append(float(disconnect_rate))
        #     Delay_Outage_Rate_list.append(float(Delay_Outage_Rate))
        #     sys_per.update_AdaptSpeed([reward_list, disconnect_rate_list, Delay_Outage_Rate_list], t)
    
        s_ = np.transpose(s_)

        state_[0,0:9,0] = state_[0,1:10,0]
        state_[0,9,0] = s_
        # print(state_)
        line_fig.update(env, r, t)
        sys_per.update(env, a, r, disconnect_rate, Avg_Delay, Delay_Outage_Rate, t, Date, path)

            
        # conv_rate.store_reward(ddpg, t, i_episode, Date, path)
        # adapt_speed.store_action(t, a)

        r = np.array([r])
        # if Date<=1007 or (Date-1007)%7>=6:
        #     if t>10:
        #         ddpg_h.store_transition(state, a_t, r, state_)
        #     # if ddpg.pointer > MEMORY_CAPACITY and ddpg.pointer%128==0:
        #     if ddpg_h.pointer > MEMORY_CAPACITY:
        #         loss_a, td_error = ddpg_h.learn()
        #         csv_loss_h.writerow([t, ddpg_h.learn_time, loss_a, td_error])

        # else:
        if t>10:
            ddpg_w.store_transition(state, a_t, r, state_)
        # if ddpg.pointer > MEMORY_CAPACITY and ddpg.pointer%128==0:
        if ddpg_w.pointer > MEMORY_CAPACITY:
            loss_a, td_error = ddpg_w.learn()
            csv_loss_w.writerow([t, ddpg_w.learn_time, loss_a, td_error]) 
            wandb.log({
                "Train/TimeSlot": t+4320*(Date-1008),
                "Train/TrainTimes": ddpg_w.learn_time,
                "Train/Loss_Actor": loss_a,
                "Train/Loss_Critic": td_error
            })        

        s = s_
        state[0,0:9,0] = state[0,1:10,0]
        state[0,9,0] = s
        # print(state)
        # ep_r += r
        a_last = a

    # var = var*0.8
    # draw number of car and INDE,draw reward
    # if i_episode == 0:
    #     continue
    line_fig.Draw_Lines(Date, i_episode, path)
    sys_per.store_xsl(path)
    # EPSILON = EPSILON*0.8 
    del env
    gc.collect()


# conv_rate.Figure_ConvergenceRate(path)

# del env
para_record.close()
# ddpg_w.save_model(path)

Date = 1012
env = envirment_PDDPG.Env(Date, path)
a_last = np.ravel(np.zeros((1,a_dim)))


for i_episode in range(1):

    print('i_episode = '+str(i_episode))
    sys_per.reset(Date, i_episode, 1)
    line_fig.reset()
    # adapt_speed.reset(SLOTNUM, a_dim)
    # conv_rate.reset(Date, path)

    for t in range(1, SLOTNUM):


        aa = ddpg_w.choose_action_2(state, env, a_last, EPSILON)
        a_t = copy.deepcopy(aa).numpy()
        a = aa[0,:1910].numpy() 


        for i in range(a_dim):
            if a[i] >= 0.5:
                a[i] = 1
            else:
                a[i] = 0


        loadrate, mixed_loadrate = env.update(a, a_last, i_episode, t+(Date-1001)*SLOTNUM)
        s_ = mixed_loadrate
        disconnect_rate, Avg_Delay, Delay_Outage_Rate = output_PDDPG_2D.Calculate_disconnect_outofdelay(env)
        r = output_PDDPG_2D.Calculate_Reward_new(a, loadrate, disconnect_rate, Delay_Outage_Rate, env, args)


        # if i_episode%3==0 and t%100==0:
        #     reward_list.append(float(r))
        #     disconnect_rate_list.append(float(disconnect_rate))
        #     Delay_Outage_Rate_list.append(float(Delay_Outage_Rate))
        #     sys_per.update_AdaptSpeed([reward_list, disconnect_rate_list, Delay_Outage_Rate_list], t)
    
        s_ = np.transpose(s_)

        state_[0,0:9,0] = state_[0,1:10,0]
        state_[0,9,0] = s_
        # print(state_)
        line_fig.update(env, r, t)
        sys_per.update(env, a, r, disconnect_rate, Avg_Delay, Delay_Outage_Rate, t, Date, path)
        # conv_rate.store_reward(ddpg, t, i_episode, Date, path)
        # adapt_speed.store_action(t, a)

        r = np.array([r])
        s = s_
        state[0,0:9,0] = state[0,1:10,0]
        state[0,9,0] = s
        # print(state)
        # ep_r += r
        a_last = a

    # draw number of car and INDE,draw reward
    # if i_episode == 0:
    #     continue
    line_fig.Draw_Lines(Date, i_episode, path)
    sys_per.store_xsl(path)
    # EPSILON = EPSILON*0.8 
    env.Reset()
    gc.collect()

del env