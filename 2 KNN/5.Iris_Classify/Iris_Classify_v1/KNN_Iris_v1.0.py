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
import math
import operator

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
        
        #从csv中读取书剑并返回行数
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
函数说明:计算距离

Parameters:
    instance1
    instance2
    length - 长度
Returns:
    距离
"""
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        #计算距离的平方和
        distance += pow((instance1[x]-instance2[x]), 2)
    return math.sqrt(distance)

"""
函数说明:回K个最近邻

Parameters:
   trainingSet - 训练街
   testInstance 
   k
Returns:
	neighbors 返回k近邻
"""
#
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        #testinstance
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
        #distances.append(dist)
    ##将邻近距离排序
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
        return neighbors

"""
函数说明:对k个近邻进行合并

Parameters:
	neighbors - k 近邻
Returns:
	value最大的key
"""
def getResponse(neighbors):
    classVotes = {}
    
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

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
        
        neighbors = getNeighbors(trainSet, testSet[x], k)
        result = getResponse(neighbors)
        
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