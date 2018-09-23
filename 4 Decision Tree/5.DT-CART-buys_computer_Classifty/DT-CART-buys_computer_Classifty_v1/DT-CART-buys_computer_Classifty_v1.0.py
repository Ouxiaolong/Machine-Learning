"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-09-16
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding: utf-8 -*-
import pandas as pd
import operator

"""
函数说明:计算基尼指数
Parameters:
	dataSet - 数据集
Returns:
	计算结果

"""
def calcGini(dataSet):
    
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: # 遍历每个实例，统计标签的频数
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    Gini = 1.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        Gini -= prob * prob # 以2为底的对数
    return Gini

"""
函数说明:计算给定特征下的基尼指数
Parameters:
	dataSet - 数据集
    feature - 特征维度
    value - 该特征变量所取的值
Returns:
	计算结果

"""
def calcGiniWithFeat(dataSet, feature, value):

    D0 = []; D1 = []
    # 根据特征划分数据
    for featVec in dataSet:
        if featVec[feature] == value:
            D0.append(featVec)
        else:
            D1.append(featVec)
    Gini = len(D0) / len(dataSet) * calcGini(D0) + len(D1) / len(dataSet) * calcGini(D1)
    return Gini

"""
函数说明:选择最优特征

Parameters:
	dataSet - 数据集
Returns:
	bestFeat - 最优特征

"""
def chooseBestSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    bestGini = 0; bestFeat = 0;newGini = 0
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        for splitVal in uniqueVals:
            newGini = calcGiniWithFeat(dataSet, i, splitVal)
            if newGini < bestGini:
                bestFeat = i
                bestGini = newGini
    return bestFeat

"""
函数说明:按照给定特征划分数据集

Parameters:
	dataSet - 待划分的数据集
	axis - 划分数据集的特征
	value - 需要返回的特征的值
Returns:
	无
"""
def splitDataSet(dataSet, axis, value):		
    #创建返回的数据集列表
    retDataSet = []										
    #遍历数据集
    for featVec in dataSet: 							
        if featVec[axis] == value:
            #去掉axis特征
            reducedFeatVec = featVec[:axis]
            #将符合条件的添加到返回的数据集
            reducedFeatVec.extend(featVec[axis+1:]) 	
            
            retDataSet.append(reducedFeatVec)
	
    #返回划分后的数据集
    return retDataSet	

"""
函数说明:统计classList中出现此处最多的元素(类标签)

Parameters:
	classList - 类标签列表
Returns:
	sortedClassCount[0][0] - 出现此处最多的元素(类标签)

"""
def majorityCnt(classList):
    classCount = {}
    
    for vote in classList:		
        #统计classList中每个元素出现的次数
        if vote not in classCount.keys():
            classCount[vote] = 0	
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)		#根据字典的值降序排序
    
    #返回classList中出现次数最多的元素
    return sortedClassCount[0][0]								

"""
函数说明:创建决策树

Parameters:
	dataSet - 训练数据集
	labels - 分类属性标签
	featLabels - 存储选择的最优特征标签
Returns:
	myTree - 决策树

"""
def createTree(dataSet, labels,featLabels):
    
    #取分类标签(是否放贷:yes or no)
    classList = [example[-1] for example in dataSet]			
    
    #如果类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):			
        return classList[0]
     
    #遍历完所有特征时返回出现次数最多的类标签
    if len(dataSet[0]) == 1:									
        return majorityCnt(classList)
    
    bestFeat= chooseBestSplit(dataSet)	#选择最优特征
    bestFeatLabel = labels[bestFeat]#最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}			#根据最优特征的标签生成树
    del(labels[bestFeat])			#删除已经使用特征标签
    
    #得到训练集中所有最优特征的属性值
    featValues = [example[bestFeat] for example in dataSet]		
    
    uniqueVals = set(featValues)		#去掉重复的属性值
    #遍历特征，创建决策树。	
    for value in uniqueVals:	
        subLabels = labels[:]
    					
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels,featLabels)

    return myTree

"""
函数说明:使用决策树分类

Parameters:
	inputTree - 已经生成的决策树
	featLabels - 存储选择的最优特征标签
	testVec - 测试数据列表，顺序对应最优特征标签
Returns:
	classLabel - 分类结果
""" 
def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))		#获取决策树结点
    secondDict = inputTree[firstStr]				#下一个字典
    featIndex = featLabels.index(firstStr)		
    
    for key in secondDict.keys():
        if testVec[featIndex] == key:

            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: 
                classLabel = secondDict[key]
    return classLabel

#测试
if __name__ == '__main__':
    ## Step 1: load data
    print("Step 1: load data...")

    df=pd.read_csv('data.csv')
    data=df.values[:-1,1:].tolist()
    
    dataSet=data[:]
    label=df.columns.values[1:-1].tolist()
    labels=label[:]
    
    #print(dataSet)
    #print(labels)
    ## Step 2: training...
    print("Step 2: training...")

    featLabels = []
    myTree = createTree(dataSet, labels,featLabels)
    #print(myTree)
    
    ## Step 3: testing
    print("Step 3: testing...")
    #测试数据
    testVec = ['middle_aged', 'yes', 'excellent', 'low']
    
    print("测试实例："+ str(testVec))
    result = classify(myTree, featLabels, testVec)
    
    ## Step 4: show the result
    print("Step 4: show the result...")
    print("result:"+ str(result))
    if result == 'yes':
        print("要购买")
    else:
        print("不购买")
