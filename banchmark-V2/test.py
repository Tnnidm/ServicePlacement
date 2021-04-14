import numpy as np
from matplotlib import pyplot as plt
plt.figure(figsize=(10,10))
n = 1
X = np.arange(n)+1
#X是1,2,3,4,5,6,7,8,柱的个数
# numpy.random.uniform(low=0.0, high=1.0, size=None), normal
#uniform均匀分布的随机数，normal是正态分布的随机数，0.5-1均匀分布的数，一共有n个
Y = [712.6650, 220.5163, 232.7826, 524.5636]
hatch = []
color = [(0.3098, 0.5059, 0.74120), (0.6078, 0.7333, 0.3490), (0.7490, 0.3137, 0.3020),\
(0.50, 0.50, 0.50), (0.93, 0.69, 0.13), (0.93, 0.69, 0.13), (0.30, 0.75, 0.93),\
(0.50, 0.39, 0.64), (0.15, 0.15, 0.15), (0.18, 0.64, 0.54)]


patterns = ['-', '+', 'x', '\\', '*', 'o', 'O', '.']


plt.bar(X,Y[0], width = 0.1, facecolor = 'white',edgecolor = color[0],hatch='/')
plt.bar(X+0.11,Y[1], width = 0.1, facecolor = 'white',edgecolor = color[1], hatch='\\')
plt.bar(X+0.11*2,Y[2], width = 0.1, facecolor = 'white',edgecolor = color[2], hatch='xx')
plt.bar(X+0.11*3,Y[3], width = 0.1, facecolor = 'white',edgecolor = color[3], hatch='x')
# plt.bar(X+0.11*4,Y5, width = 0.1, facecolor = 'white',edgecolor = 'purple', hatch='//')
# plt.bar(X+0.11*5,Y6, width = 0.1, facecolor = 'white',edgecolor = 'grey', hatch='\\\\')


#水平柱状图plt.barh，属性中宽度width变成了高度height
#打两组数据时用+
#facecolor柱状图里填充的颜色
#edgecolor是边框的颜色
#想把一组数据打到下边，在数据前使用负号
#plt.bar(X, -Y2, width=width, facecolor='#ff9999', edgecolor='white')
#给图加text
plt.legend(['p=0.99,l=1,g=1','p=0.9,l=1,g=1','p=0.99,l=5,g=1','p=0.99,l=1,g=5'], loc=9)
# for x,y in zip(X,Y1):
#     plt.text(x+0.3, y+0.05, '%.2f' % y, ha='center', va= 'bottom')

# for x,y in zip(X,Y2):
#     plt.text(x+0.6, y+0.05, '%.2f' % y, ha='center', va= 'bottom')
plt.ylim(0,800)

plt.show()

import numpy as np
from matplotlib import pyplot as plt
x = 1