"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-09-16
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding: utf-8 -*-
import numpy as np
from functools import reduce

"""
函数说明:创建实验样本

Parameters:
	无
Returns:
	postingList - 实验样本切分的词条
	classVec - 类别标签向量
"""
def loadDataSet():#实验样本集，返回文档集合和类别标签，人工手动标注
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],				#切分的词条
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	
    #类别标签向量，1代表低俗性词汇，0代表不是
    classVec = [0,1,0,1,0,1]   																
   
    #返回实验样本切分的词条和类别标签向量 
    return postingList,classVec																

"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

Parameters:
	dataSet - 整理的样本数据集
Returns:
	vocabSet - 返回不重复的词条列表，也就是词汇表
"""
#创建一个包含在所有文档中出现的不重复的列表 |用于求两个集合并集，词集
def createVocabList(dataSet):
    #创建一个空的不重复列表
    vocabSet = set([]) 
    #遍历数据集
    for document in dataSet:				
        vocabSet = vocabSet | set(document) #取并集
    #print(vocabSet)	
    return list(vocabSet)#生成一个包含所有单词的列表

"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

Parameters:
	vocabList - createVocabList返回的列表
	inputSet - 切分的词条列表
Returns:
	returnVec - 文档向量,词集模型
"""
def setOfWords2Vec(vocabList, inputSet):
    #创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)									
    
    #遍历每个词条
    for word in inputSet:			
        #如果词条存在于词汇表中，则置1									
        if word in vocabList:											
            returnVec[vocabList.index(word)] = 1
        else: 
            print("the word: %s is not in my Vocabulary!" % word)
    #print(returnVec)
    return returnVec		#返回文档向量											


"""
函数说明:朴素贝叶斯分类器训练函数

Parameters:
	trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
	trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
	p0Vect - 非低俗的条件概率数组
	p1Vect - 低俗类的条件概率数组
	pAbusive - 文档属于低俗类的概率

"""
def trainNB(trainMatrix,trainCategory):

    #计算训练的文档条数
    numTrainDocs = len(trainMatrix)							
    #print("numTrainDocs:" + str(numTrainDocs))
    
    #计算每篇文档的词条数
    numWords = len(trainMatrix[0])							
    #print("numWords:" + str(numWords))
    
    #文档属于低俗类的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)	
    
    #创建numpy.zeros数组,
    p0Num = np.zeros(numWords); 
    p1Num = np.zeros(numWords)	
    
    #分母初始化为0.0
    p0Denom = 0.0; 
    p1Denom = 0.0                        	
    
    for i in range(numTrainDocs):
        #统计属于低俗类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
        if trainCategory[i] == 1:							
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:												
            #统计属于非低俗类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    
    p1Vect = p1Num/p1Denom									#相除        
    p0Vect = p0Num/p0Denom   
    
    #print(p1Num)
    #print(p0Num)
    #print(p1Denom)
    #print(p0Denom)
      
    return p0Vect,p1Vect,pAbusive		#返回属于低俗类的条件概率数组，属于非低俗类的条件概率数组，文档属于低俗类的概率

"""
函数说明:朴素贝叶斯分类器分类函数

Parameters:
	vec2Classify - 待分类的词条数组
	p0Vec - 非低俗类的条件概率数组
	p1Vec -低俗类的条件概率数组
	pClass1 - 文档属于低俗类的概率
Returns:
	0 - 属于非低俗类
	1 - 属于低俗类

"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):

    #从左到右对一个序列的项累计地应用有两个参数的函数，以此合并序列到一个单一值。
    #例如，reduce(lambda x, y: x+y, [1, 2, 3, 4, 5])  计算的就是((((1+2)+3)+4)+5)。
    p1 = reduce(lambda x,y:x*y, vec2Classify * p1Vec) * pClass1    			#对应元素相乘
    p0 = reduce(lambda x,y:x*y, vec2Classify * p0Vec) * (1.0 - pClass1)
	
    print('p0:',p0)
    print('p1:',p1)
    
    if p1 > p0:
        return 1
    else: 
        return 0

"""
函数说明:测试朴素贝叶斯分类器

Parameters:
	无
Returns:
	无
"""
def testingNB():
    
	## Step 1: load data
    print("Step 1: load data...")
    #创建数据
    #创建实验样本
    listPosts,listClasses = loadDataSet()									
    #创建词汇表
    myVocabList = createVocabList(listPosts)

    #向量化样本
    trainMat=[]
    for postinDoc in listPosts: 
        #将实验样本向量化
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))				
    #print(trainMat)
    
	## Step 2: training...
    print("Step 2: training...")
    #训练朴素贝叶斯分类器
    p0V,p1V,pAb = trainNB(np.array(trainMat),np.array(listClasses))		
    
	## Step 3: testing
    print("Step 3: testing...")
    #测试样本1
    testEntry = ['love', 'my', 'dalmation']									
    
    #测试样本向量化
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))				
    
	## Step 4: show the result
    print("Step 4: show the result...")
	if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于低俗类')										
        #执行分类并打印分类结果
    else:
        print(testEntry,'属于非低俗类')										
        #执行分类并打印分类结果

    #测试样本2
    testEntry = ['stupid', 'garbage']										
    #测试样本向量化
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))				
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于低俗类')										
        #执行分类并打印分类结果
    else:
        print(testEntry,'属于非低俗类')										
        #执行分类并打印分类结果

if __name__ == '__main__':
	testingNB()
