"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-09-08
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding: utf-8 -*-
import csv#用于处理csv文件
import random#用于随机数
import operator
import numpy as np

"""
函数说明:加载数据

Parameters:
  filename - 文件名
  split - 分隔符
  trainingSet - 训练集
  testSet - 测试集
Returns:
	无
"""
def loadDataset(filename, split, trainSet = [], testSet = []):
    with open(filename, 'rt') as csvfile:
        
        #从csv中读取数据并返回行数
        lines = csv.reader(csvfile)
        
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            #保存数据集到训练集和测试集#random.random()返回随机浮点数
            if random.random() < split:
                trainSet.append(dataset[x])
            else:
                #将获得的测试数据放入测试集中
                testSet.append(dataset[x])

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
def classify(inX, dataSet, labels, k):
	#numpy函数shape[0]返回dataSet的行数
	dataSetSize = dataSet.shape[0]
	
	#在列向量方向上重复inX共1次(横向),行向量方向上重复inX共dataSetSize次(纵向)
	diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
	
	#二维特征相减后平方
	sqDiffMat = diffMat**2
	
	#sum()所有元素相加,sum(0)列相加,sum(1)行相加
	sqDistances = sqDiffMat.sum(axis=1)
	
	#开方,计算出距离
	distances = sqDistances**0.5
	
	#返回distances中元素从小到大排序后的索引值
	sortedDistIndices = distances.argsort()
	
	#定一个记录类别次数的字典
	classCount = {}
	for i in range(k):
		#取出前k个元素的类别
		voteIlabel = labels[sortedDistIndices[i]]
		#dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
		#计算类别次数
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	#python3中用items()替换python2中的iteritems()
	#key=operator.itemgetter(1)根据字典的值进行排序
	#key=operator.itemgetter(0)根据字典的键进行排序
	#reverse降序排序字典
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	
	#返回次数最多的类别,即所要分类的类别
	return sortedClassCount[0][0]

"""
函数说明:计算准确率

Parameters:
  testSet - 测试集
  predictions - 预测值 
Returns:
	返回准确率
"""
#计算准确率
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0

"""
函数说明:分割数据

Parameters:
    dataSet - 数据集
Returns:
    data_X - 特征数据集
    data_Y - 标签数据集
"""
def segmentation_Data(dataSet):
    
    #得到文件行数
    Lines = len(dataSet)
    
    #返回的NumPy矩阵,解析完成的数据:4列
    data_X = np.zeros((Lines,4))
    data_Y = []
    for x in range(Lines):
        data_X[x,:] = dataSet[x][0:4]
        data_Y.append(dataSet[x][-1])
    
    return data_X, data_Y


"""
函数说明:主函数

Parameters:
	无
Returns:
	无
"""
def main():
    #prepare data
    trainSet = []#训练数据集
    testSet = []#测试数据集
    split = 0.67#分割的比例
    
    ## step 1: load data
    #加载数据集
    print("step 1: load data...")
    loadDataset('C:/TensorFlow/irisdata.txt', split, trainSet, testSet)
    
    #数据集分割
    train_X,train_Y = segmentation_Data(trainSet)
    test_X,test_Y = segmentation_Data(testSet)
    
    print('Train set: ' + repr(len(trainSet)))
    print('Test set: ' + repr(len(testSet)))
    
    #print(train_X)
    #print(train_Y)
 
    ## step 2: training...
    print("step 2: training...")
    pass

    #generate predictions
    predictions = []
    k = 3
	
    ## step 3: testing
    print("step 3: testing...")
    for x in range(len(testSet)):
        
        result = classify(test_X[x], train_X, train_Y, k)
        
        predictions.append(result)
        #print ('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]) + "\n")
    
    #print('predictions: ' + repr(predictions))
    
    ## step 4: show the result
    print("step 4: show the result...")    
    
    #准确率
    accuracy = getAccuracy(testSet, predictions)
    print('\nAccuracy: ' + repr(accuracy) + '%')

if __name__ == '__main__':
    main()