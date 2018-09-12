"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-09-08
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding: utf-8 -*-
import math

def ComputeEuclideanDistance(x1, y1, x2, y2):
    d = math.sqrt(math.pow((x1-x2), 2) + math.pow((y1-y2), 2))
    return d

#计算距离
d_ag1 = ComputeEuclideanDistance(3, 104, 18, 90)
d_ag2 = ComputeEuclideanDistance(2, 100, 18, 90)
d_ag3 = ComputeEuclideanDistance(1, 81, 18, 90)
d_ag4 = ComputeEuclideanDistance(101, 10, 18, 90)
d_ag5 = ComputeEuclideanDistance(99, 5, 18, 90)
d_ag6 = ComputeEuclideanDistance(88, 12, 18, 90)

#打印数据
print(d_ag1)
print(d_ag2)
print(d_ag3)
print(d_ag4)
print(d_ag5)
print(d_ag6)


