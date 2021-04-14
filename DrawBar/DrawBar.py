import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
# print(mpl.__version__)
from matplotlib.backends.backend_pdf import PdfPages
# print(mpl.get_cachedir())
pdf = PdfPages('figure1.pdf')
# import matplotlib.font_manager
# matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')


mpl.rcParams["font.family"] = 'Helvetica'

x = 0.2 #请勿修改，和比例有关

X = 1 # 起始坐标
XX = 1.6 # 间隔
w = 1 # 柱子宽度
lw = 2 # 线宽
color = [(0.3098, 0.5059, 0.74120), (0.6078, 0.7333, 0.3490), (0.7490, 0.3137, 0.3020),\
        (0.50, 0.50, 0.50), (0.93, 0.69, 0.13),  (0.30, 0.75, 0.93),\
        (0.50, 0.39, 0.64), (0.15, 0.15, 0.15), (0.18, 0.64, 0.54)] # RGB色彩

FontSize1 = 18 # 小的字体
FontSize2 = 26 # 大的字体

NO = 4 # 第几个图
if NO ==  1:
    Y_Label = 'Average system utility'
    Y = [712.6650, 220.5163, 232.7826, 524.5636]
    legend_list = ['p=0.99,l=1,g=1','p=0.9,l=1,g=1','p=0.99,l=5,g=1','p=0.99,l=1,g=5']
    Y_limid = (0, 800)
    lo = 9 #位置
    nco = 1 #legend分成几列
    fig = plt.figure(figsize=(5.2*(1/(1-x)),5.2), dpi = 100)
elif NO == 2:
    Y_Label = 'Average resource utilization'
    Y = [0.6600, 0.4188, 0.4270, 0.5380]
    legend_list = ['p=0.99,l=1,g=1','p=0.9,l=1,g=1','p=0.99,l=5,g=1','p=0.99,l=1,g=5']
    Y_limid = (0, 0.7)
    lo = 9 #位置
    nco = 1 #legend分成几列

    fig = plt.figure(figsize=(5.2*(1/(1-x)),5.2), dpi = 100)
elif NO == 3:
    Y_Label = 'Average system'+'\n'+'utility'
    Y = [712.6650, 170.0032, 264.8931, -426.2866, 681.1589, 638.6501271, -591.4337, 841.8514, -2115]
    legend_list = ['DeepReserve','DC','DA','DDPG','UC','HAF','GSP','Opt','Rand']
    Y_limid = (-2000,1000)
    lo = 3 #位置
    nco = 3 #legend分成几列
    fig = plt.figure(figsize=(2*5.2*(1/(1-x)),2*5.2/2.541), dpi = 100)
    
elif NO == 4:
    Y_Label = 'Average resource'+'\n'+'utilization'
    Y = [0.6600, 0.4671, 0.4824, 0.3258, 0.6259, 0.5990, 0.2936, 0.7804, 0.1576]
    legend_list = ['DeepReserve','DC','DA','DDPG','UC','HAF','GSP','Opt','Rand']
    Y_limid = (0,0.8)
    lo = 3 #位置
    nco = 3 #legend分成几列
    fig = plt.figure(figsize=(2*5.2*(1/(1-x)),2*5.2/2.541), dpi = 100)
else:
    print('Input an error NO!')
    exit()
pdf = PdfPages('figure'+str(NO)+'.pdf')

patterns = ['/', '\\', 'xx', 'x', '\\\\', '//', '+', '..', '++']

mpl.rcParams['hatch.linewidth'] = lw
for i in range(len(Y)):
    plt.bar(X+i*XX,Y[i], width = w, facecolor = 'white',edgecolor = color[i], hatch=patterns[i], linewidth=lw)



plt.rc('legend', fontsize=FontSize1)
# plt.rc('legend', font='Helvetica')
plt.legend(legend_list, loc=lo, fancybox = False, edgecolor='black', borderpad = 0.2, labelspacing = 0.2, handletextpad = 0.3, ncol = nco)
if NO == 3:
    plt.axhline(y=0, color = 'black', linewidth=0.8)
list1 = []
list2 = []
for i in range(len(Y)):
    list1.append(1+i*XX)
    list2.append(' ')
plt.xticks(list1, list2, fontsize=FontSize1)
plt.yticks(fontsize=FontSize1)
plt.xlabel(" ",fontsize=FontSize2)
plt.ylabel(Y_Label,fontsize=FontSize2)

# for x,y in zip(X,Y1):
#     plt.text(x+0.3, y+0.05, '%.2f' % y, ha='center', va= 'bottom')

# for x,y in zip(X,Y2):
#     plt.text(x+0.6, y+0.05, '%.2f' % y, ha='center', va= 'bottom')
plt.ylim(Y_limid)
# pdf.savefig()

plt.subplots_adjust(left=x)
# plt.tight_layout()
# fig.savefig('figure1.eps',format='eps')
pdf.savefig()
plt.show()

pdf.close()