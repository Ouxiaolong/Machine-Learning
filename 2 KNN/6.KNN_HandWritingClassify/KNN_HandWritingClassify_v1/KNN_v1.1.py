"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-09-08
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding: UTF-8 -*-
import numpy as np
import operator
import os

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
函数说明:将32x32的二进制图像转换为1x1024向量。

Parameters:filename - 文件名
Returns:returnVect - 返回的二进制图像的1x1024向量
"""
def img2vector(filename):
    rows = 32
    cols = 32
    
    #创建1x1024零向量
    imgVector = np.zeros((1, rows * cols))
	
    #打开文件,读取每行内容
    fr = open(filename)
	
    #按行读取
    for r in range(rows):
        #读一行数据
        lineStr = fr.readline()
		
        #每一行的前32个元素依次添加到returnVect中
        for c in range(cols):
            imgVector[0, rows * r + c] = int(lineStr[c])
	
    #返回转换后的1x1024向量
    return imgVector

"""
函数说明:加载数据

Parameters:
	无
Returns:
	train_x - 训练集样本
	train_y - 训练集对应的数字标签
	test_x - 测试集数据
	test_y - 测试集对应的数字标签
"""
def loadDataSet():
    ## step 1: Getting training set
    print("---Getting training set...")
    
    dataSetDir = 'C:/TensorFlow/'
    #返回trainingDigits目录下的文件名
    trainingFileList = os.listdir(dataSetDir + 'trainingDigits') # load the training set
    
    #返回文件夹下文件的个数
    numSamples = len(trainingFileList)
 
    # 初始化样本数据矩阵（numSamples*1024）
    train_x = np.zeros((numSamples, 1024))
    train_y = []
    
    #从文件名中解析出训练集的类别
    for i in range(numSamples):
        #获得文件的名字
        filename = trainingFileList[i]
        
        ##将每一个文件的1x1024数据存储到train_x矩阵中
        train_x[i, :] = img2vector(dataSetDir + 'trainingDigits/%s' % filename) 
        
        #获得分类的数字，也就是分类标签
        label = int(filename.split('_')[0]) # return 1
        #将获得的类别添加到train_y中
        train_y.append(label)
 
    ## step 2: Getting testing set
    print("---Getting testing set...")
    #返回testDigits目录下的文件名
    testingFileList = os.listdir(dataSetDir + 'testDigits') # load the testing set
    
    #返回文件夹下文件的个数
    numSamples = len(testingFileList)
    
    # 初始化测试样本数据矩阵（numSamples*1024）
    test_x = np.zeros((numSamples, 1024))
    test_y = []
    
    for i in range(numSamples):
        #获得文件的名字
        filename = testingFileList[i]
 
        #将每一个文件的1x1024数据存储到test_x矩阵中
        test_x[i, :] = img2vector(dataSetDir + 'testDigits/%s' % filename) 
 
        #获得分类的数字，也就是分类标签
        label = int(filename.split('_')[0]) # return 1
        #将获得的类别添加到test_y中
        test_y.append(label)
 
    return train_x, train_y, test_x, test_y

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
        if testSet[x] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0 , correct

"""
函数说明:手写数字分类测试

Parameters:
	无
Returns:
	无
"""
def testHandWritingClass():
    ## step 1: load data
    print("step 1: load data...")
    train_x, train_y, test_x, test_y = loadDataSet()
 
    ## step 2: training...
    print("step 2: training...")
    pass
 
    ## step 3: testing
    print("step 3: testing...")
    numTestSamples = test_x.shape[0]
    matchCount = 0
    #generate predictions
    predictions = []
    
    for i in range(numTestSamples):
        predict = classify(test_x[i], train_x, train_y, 3)
        predictions.append(predict)
        #print("Really Lable: %d \t KNN Lable :%d" % (test_y[i],predict))
        
    accuracy,matchCount = getAccuracy(test_y, predictions)
 
    ## step 4: show the result
    print("step 4: show the result...")    
    print("总共错了%d个数据\n" % (numTestSamples-matchCount))
    print('准确率是: %.2f%%' % (accuracy))

if __name__ == '__main__':
	testHandWritingClass()
