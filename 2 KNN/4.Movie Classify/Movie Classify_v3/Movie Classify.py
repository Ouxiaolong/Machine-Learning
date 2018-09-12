"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-09-08
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import neighbors 

"""
函数说明:创建数据集
Parameters:
    无
Returns:
    group - 数据集
    labels - 分类标签
"""
def createDataSet():
    #四组二维特征
    group = np.array([[3,104],[2,100],[1,81],[101,10],[99,5],[88,2]])
    #四组特征的标签
    labels = ['爱情片','爱情片','爱情片','动作片','动作片','动作片']
    return group, labels

#第一步：取得knn分类器  
knn = neighbors.KNeighborsClassifier() 

#第二步：创建数据
data , lables = createDataSet()

#第三步：训练数据
knn.fit(data,lables) #导入数据进行训练 

#第四步：预测数据
print(knn.predict([[18,90]]))
