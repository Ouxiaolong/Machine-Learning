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
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

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

#############################可视化##############################
"""
函数说明:获取决策树叶子结点的数目

Parameters:
    myTree - 决策树
Returns:
    numLeafs - 决策树的叶子结点的数目
"""
def getNumLeafs(myTree):
    #初始化叶子
    numLeafs = 0                                               
        
    #python3中myTree.keys()返回的是dict_keys,
    #不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，
    #可以使用list(myTree.keys())[0]
    firstStr = next(iter(myTree))                                

    #获取下一组字典
    secondDict = myTree[firstStr]                                
    
    for key in secondDict.keys():
        #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__=='dict':               
           
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

"""
函数说明:获取决策树的层数

Parameters:
    myTree - 决策树
Returns:
    maxDepth - 决策树的层数
"""
def getTreeDepth(myTree):
    #初始化决策树深度
    maxDepth = 0                                                
    
    #python3中myTree.keys()返回的是dict_keys,
    #不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，
    #可以使用list(myTree.keys())[0]
    firstStr = next(iter(myTree))                                
    
    #获取下一个字典
    secondDict = myTree[firstStr]                                
    
    for key in secondDict.keys():
        #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__=='dict':                
            
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   
            thisDepth = 1
        if thisDepth > maxDepth: 
            maxDepth = thisDepth   #更新层数
    return maxDepth

"""
函数说明:绘制结点

Parameters:
    nodeTxt - 结点名
    centerPt - 文本位置
    parentPt - 标注的箭头位置
    nodeType - 结点格式
Returns:
    无
"""
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")     #定义箭头格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  #设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',    #绘制结点
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)

"""
函数说明:标注有向边属性值

Parameters:
    cntrPt、parentPt - 用于计算标注位置
    txtString - 标注的内容
Returns:
    无
"""
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]     #计算标注位置                   
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

"""
函数说明:绘制决策树

Parameters:
    myTree - 决策树(字典)
    parentPt - 标注的内容
    nodeTxt - 结点名
Returns:
    无
"""
def plotTree(myTree, parentPt, nodeTxt):
    #设置结点格式
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")     
    
    #设置叶结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")     
   
    #获取决策树叶结点数目，决定了树的宽度
    numLeafs = getNumLeafs(myTree)               
    
    depth = getTreeDepth(myTree)   #获取决策树层数
    firstStr = next(iter(myTree))                 #下个字典                                                 
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)    #中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)         #标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)     #绘制结点
    secondDict = myTree[firstStr]             #下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD               #y偏移
    
    for key in secondDict.keys():                               
        if type(secondDict[key]).__name__=='dict':       #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key],cntrPt,str(key))            #不是叶结点，递归调用继续绘制
        else:                                           #如果是叶结点，绘制叶结点，并标注有向边属性值                                             
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

"""
函数说明:创建绘制面板

Parameters:
    inTree - 决策树(字典)
Returns:
    无
"""
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')                   #创建fig
    fig.clf()                                               #清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))              #获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))             #获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;       #x偏移
    plotTree(inTree, (0.5,1.0), '')                                 #绘制决策树
    plt.show()

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
    myTree = createTree(dataSet, labels, featLabels)
    #print(myTree)

    ## Step 3: show pic
    print("Step 3: show the picture...")
    createPlot(myTree)
    
    ## Step 4: testing
    print("Step 4: testing...")
    #测试数据
    testVec = ['middle_aged', 'yes', 'excellent', 'low']
    
    print("测试实例："+ str(testVec))
    result = classify(myTree, featLabels, testVec)
    
    ## Step 5: show the result
    print("Step 5: show the result...")
    print("result:"+ str(result))
    if result == 'yes':
        print("要购买")
    else:
        print("不购买")
   