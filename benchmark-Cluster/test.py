import numpy as np
# import PDDPG_2D

# ddpg_w = PDDPG_2D.PDDPG(1910, 1910)

# state = np.ones((1, 10, 1, 1, 1910))

# aa = ddpg_w.choose_action_3(state, 0.97)


# # state[0,7,0] = np.zeros((1,1910))
# state[0,8,0] = np.zeros((1,1910))
# state[0,9,0] = np.zeros((1,1910))

# aa = ddpg_w.choose_action_3(state, 0.97)



a = np.random.rand(5,5)
print(a)
b = a[:,[1,3]]
print(b)