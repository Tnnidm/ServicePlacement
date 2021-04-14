import torch



'''
调的参数
'''
CONVLSTM_LAYERS = 5
EPSILON = 0.97
L1_INDE_CAPACITY = 5 # how many cars one INDE can serve 
IDLE_TIMES = 1




'''
Parameters about DDPG
'''
MEMORY_CAPACITY = 4320
LR_A = 0.001    # learning rate for actor
LR_C = 0.0005    # learning rate for critic
# LR_A = 0.01    # learning rate for actor
# LR_C = 0.02    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
BATCH_SIZE = 32
use_gpu = torch.cuda.is_available()

'''
Parameters about environment
'''
L = 30 # the numbers of rows and columns
MAX_QUEUING_TIME = 1000

L2_INDE_CAPACITY = 10
Bt_INDE_num = 1910


'''
control mode
'''
PrintLogType = -1
PictureType = -1


'''
action and state
'''
a_dim = 1910
s_dim = 1910


'''
train mode
'''
MAX_EPISODES = 4
SLOTNUM = 4320 # the numbers of slots used


'''
Parameters about loss
'''
# about utilization ratio punishment
U_BEST = 1 # as the name say
PHI_1 = 1 # means the punishment of utilization ratio when u<u_best
PHI_2 = 1 #means the punishment of utilization ratio when u>u_best
# about INDE cost
ALPHA = 3 # means the cost of maintaining a basic station INDE
BETA = 1 # means the cost of maintaining a car INDE
# about centent loss punishment
OMEGA = 1 # means the punishment of centent loss

# Location
LogLocation = 'log/'
preprocessed_data_Location = 'preprocessed_data/'