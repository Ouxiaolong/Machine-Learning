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
import operator
import collections

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

"""
函数说明:kNN算法,分类器
Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
"""
def classification(inX, dataSet, labels, k):
    #第一步：计算距离
    """
    inX 是输入的测试样本，是一个[x, y]样式的
    dataset 是训练样本集
    labels 是训练样本标签
    k 是top k最相近的
    """
	#numpy函数shape[0]返回dataSet的行数
	# shape返回矩阵的[行数，列数]，
    # 那么shape[0]获取数据集的行数，
    # 行数就是样本的数量
    dataSetSize = dataSet.shape[0]
    
	#在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
    #这是和二维特征相减得到差值
    #array([[18,90],[18,90],[18,90],[18,90],[18,90],[18,90]])
    #array([[3,104],[2,100],[1,81],[101,10],[99,5],[88,2]])
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    
	#二维特征相减后平方
	# diffMat就是输入样本与每个训练样本的差值，然后对其每个x和y的差值进行平方运算。
    # diffMat是一个矩阵，矩阵**2表示对矩阵中的每个元素进行**2操作，即平方。
    sqDiffMat = diffMat**2
    
    #sum()所有元素相加，sum(0)列相加，sum(1)行相加
	# axis=1表示按照横轴，sum表示累加，即按照行进行累加。
    sqDistances = sqDiffMat.sum(axis=1)
    
	#开方，计算出距离
    distances = sqDistances**0.5
    
    #第二步：距离排序
    #返回distances中元素从小到大排序后的索引值
	# 按照升序进行快速排序，返回的是原数组的下标。
    # 比如，x = [30, 10, 20, 40]
    # 升序排序后应该是[10,20,30,40],他们的原下标是[1,2,0,3]
    # 那么，numpy.argsort(x) = [1, 2, 0, 3]
    sortedDistIndices = distances.argsort()
    
    #第三步：分类
    #定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        #取出前k个元素的类别# index = sortedDistIndicies[i]是第i个最相近的样本下标
        # voteIlabel = labels[index]是样本index对应的分类结果('爱情片' or '动作片'
        voteIlabel = labels[sortedDistIndices[i]]
        #dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        
		#计算类别次数
		# classCount.get(voteIlabel, 0)返回voteIlabel的值，如果不存在，则返回0
        # 然后将票数增1
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #python3中用items()替换python2中的iteritems()
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    #reverse降序排序字典
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

def classification_new(inx, dataset, labels, k):
	# 计算距离
	dist = np.sum((inx - dataset)**2, axis=1)**0.5
	# k个最近的标签
	k_labels = [labels[index] for index in dist.argsort()[0 : k]]
	# 出现次数最多的标签即为最终类别
	label = collections.Counter(k_labels).most_common(1)[0][0]
	return label

if __name__ == '__main__':
    #创建数据集
    group, labels = createDataSet()
    #测试集
    test = [18,90]
    #kNN分类
    #test_class = classification(test, group, labels, 3)
    test_class = classification_new(test, group, labels, 3)
    #打印分类结果
    print(test_class)