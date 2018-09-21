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
from math import log
import operator

"""
函数说明:创建测试数据集

Parameters:
	无
Returns:
	dataSet - 数据集
	labels - 特征标签
"""
def createDataSet():
    #数据集
    dataSet = [[0, 0, 0, 2, 'no'], 
               [0, 0, 1, 2, 'no'], 
               [1, 0, 0, 2, 'yes'], 
               [2, 0, 0, 1, 'yes'],
               [2, 1, 0, 0, 'yes'], 
               [2, 1, 1, 0, 'no'], 
               [1, 1, 1, 0, 'yes'], 
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 0, 'yes'],
               [2, 1, 0, 1, 'yes'], 
               [0, 1, 1, 1, 'yes'], 
               [1, 0, 1, 1, 'yes'], 
               [1, 1, 0, 2, 'yes'],
               [2, 0, 2, 2, 'no']]
    #特征标签
    labels = ['age', 'student', 'credit_rating', 'income']

    #返回数据集和分类属性
    return dataSet, labels 	

"""
函数说明:计算给定数据集的经验熵(香农熵)

Parameters:
    dataSet - 数据集
Returns:
    shannonEnt - 经验熵(香农熵)
"""
def calcShannonEnt(dataSet):
    #返回数据集的行数
    numEntires = len(dataSet)                        
    
    #保存每个标签(Label)出现次数的字典
    labelCounts = {}                                
    
    #对每组特征向量进行统计
    for featVec in dataSet:                            
        #提取标签(Label)信息
        currentLabel = featVec[-1]                    
        
        #如果标签(Label)没有放入统计次数的字典,添加进去
        if currentLabel not in labelCounts.keys():    
            
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1   #Label计数            
        
    shannonEnt = 0.0   #经验熵(香农熵)                             
    
    #计算香农熵
    for key in labelCounts:                            
        
        #选择该标签(Label)的概率
        prob = float(labelCounts[key]) / numEntires    
        
        #利用公式计算
        shannonEnt -= prob * log(prob, 2)            
        
    #返回经验熵(香农熵)
    return shannonEnt                                						

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
函数说明:计算X_i给定的条件下，Y的条件熵

Parameters:
    dataSet - 数据集
    i - 维度i
    uniqueVals - 数据集特征集合
Returns:
    newEntropy - 条件熵
"""
def calcConditionalEntropy(dataSet, i, uniqueVals):
    
    #经验条件熵
    newEntropy = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, i, value)
        prob = len(subDataSet) / float(len(dataSet))  # 极大似然估计概率
        newEntropy += prob * calcShannonEnt(subDataSet)  # 条件熵的计算
    return newEntropy

"""
函数说明:计算信息增益

Parameters:
    dataSet - 数据集
    baseEntropy - 数据集的信息熵
Returns:
    bestIndex - 最好的特征索引
    bestInfoGain - 最好的信息增益
"""
def calcInformationGain(dataSet):

    #最优特征的索引值	
    bestIndex = -1
    #信息增益
    bestInfoGain = 0.0  	
    
    baseEntropy = calcShannonEnt(dataSet)
    
    #特征数量
    numFeatures = len(dataSet[0]) - 1	
    #遍历所有特征
    for i in range(numFeatures): 						
        #获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        #创建set集合{},元素不可重复
        uniqueVals = set(featList)     					
		
        #经验条件熵
        newEntropy = 0.0
        #计算条件熵
        newEntropy = calcConditionalEntropy(dataSet, i, uniqueVals)
        #得到增益
        infoGain = baseEntropy - newEntropy  # 信息增益，就yes熵的减少，也就yes不确定性的减少
        
        #最优增益选择
        if (infoGain > bestInfoGain): 	
            #更新信息增益，找到最大的信息增益
            bestInfoGain = infoGain 		
			
            #记录信息增益最大的特征的索引值
            bestIndex = i 
    
    return bestIndex, bestInfoGain

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
def createTree(dataSet, labels, featLabels):
    
    #取分类标签(是否放贷:yes or no)
    classList = [example[-1] for example in dataSet]			
    
    #如果类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):			
        return classList[0]
     
    #遍历完所有特征时返回出现次数最多的类标签
    if len(dataSet[0]) == 1:									
        return majorityCnt(classList)
    
    bestFeat, bestInfoGain= calcInformationGain(dataSet)	#选择最优特征
    bestFeatLabel = labels[bestFeat]#最优特征的标签
    
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}			#根据最优特征的标签生成树
    del(labels[bestFeat])			#删除已经使用特征标签
    
    #得到训练集中所有最优特征的属性值
    featValues = [example[bestFeat] for example in dataSet]		
    
    uniqueVals = set(featValues)		#去掉重复的属性值
    
    for value in uniqueVals:	#遍历特征，创建决策树。						
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)

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

    #方式一
    #dataSet, labels = createDataSet()
    
        
    #方式二
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
    myTree = createTree(dataSet, labels, featLabels)
    
    #print(myTree)
    
    ## Step 3: testing
    print("Step 3: testing...")
    #testVec = [0,0,0,2]										#测试数据
    testVec = [1,1,0,2]
    
    print("测试实例："+ str(testVec))
    result = classify(myTree, featLabels, testVec)

    ## Step 4: show the result
    print("Step 4: show the result...")
    print("result:"+ str(result))
    if result == 'yes':
        print("要购买")
    else:
        print("不购买")
        