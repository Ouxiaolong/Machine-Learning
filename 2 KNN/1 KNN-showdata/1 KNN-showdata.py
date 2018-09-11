"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-09-08
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np

#https://blog.csdn.net/treasuresss/article/details/50750325
#解决python matplotlib中文乱码问题
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 定义七个点的坐标
a1 = np.array([3, 104])
a2 = np.array([2, 100])
a3 = np.array([1,81])
b1 = np.array([101, 10])
b2 = np.array([99, 5])
b3 = np.array([88,2])
c = np.array([18,90])

# 七个点坐标分别赋值给X,Y
X1, Y1 = a1
X2, Y2 = a2
X3, Y3 = a3
X4, Y4 = b1
X5, Y5 = b2
X6, Y6 = b3
X7, Y7 = c

plt.title('电影分类')
plt.xlabel('打斗镜头')
plt.ylabel('接吻镜头')
plt.scatter(X1, Y1, color="blue", label="爱情片")
plt.scatter(X2, Y2, color="blue", label="爱情片")
plt.scatter(X3, Y3, color="blue", label="爱情片")

plt.scatter(X4, Y4, color="red", label="动作片")
plt.scatter(X5, Y5, color="red", label="动作片")
plt.scatter(X6, Y6, color="red", label="动作片")

plt.scatter(X7, Y7, color="yellow", label="未知")
plt.legend(loc='upper right')

plt.annotate(r'a(3,104)', xy=(X1, Y1), xycoords='data', xytext=(+15, +40), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.annotate(r'a2(2,100)', xy=(X2, Y2), xycoords='data', xytext=(+20, +10), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.annotate(r'a3(1,81)', xy=(X3, Y3), xycoords='data', xytext=(+15, +10), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.annotate(r'b1(101,10)', xy=(X4, Y4), xycoords='data', xytext=(+30, +10), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.annotate(r'b2(99,5)', xy=(X5, Y5), xycoords='data', xytext=(+20, +40), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.annotate(r'b3(88,2)', xy=(X6, Y6), xycoords='data', xytext=(+10, +60), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.annotate(r'c(18,90)', xy=(X7, Y7), xycoords='data', xytext=(+10, +20), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.show()



