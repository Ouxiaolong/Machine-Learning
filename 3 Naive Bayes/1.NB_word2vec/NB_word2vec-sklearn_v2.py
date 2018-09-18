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
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

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
函数说明:测试朴素贝叶斯分类器

Parameters:
	无
Returns:
	无
"""
def testingNB():
    ## Step 1: load data
    print("Step 1: load data...")
    #创建数据#创建实验样本
    listPosts,listClasses = loadDataSet()									
    #创建词汇表
    myVocabList = createVocabList(listPosts)

    #向量化样本
    trainMat=[]
    
    for postinDoc in listPosts: 
        #将实验样本向量化
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))				
    #print(trainMat)
    
    ## Step 3: init NB
    print("Step 3: init NB...")
    #初始化贝叶斯分类器
    gnb = GaussianNB()
 
    ## Step 4: training...
    print("Step 4: training...")
    #训练数据
    gnb.fit(trainMat, listClasses)

    ## Step 5: testing
    print("Step 5: testing...")
    testEntry = ['love', 'my', 'dalmation']
    #testEntry = ['stupid', 'garbage']	
    
    #测试样本向量化
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))	
    #print(thisDoc)

    #预测数据
    predictedLabel =  gnb.predict([thisDoc])
    #predictedLabel = gnb.fit(trainMat, listClasses).predict(thisDoc)

    ## Step 6: show the result
    print("Step 6: show the result...")
    print(predictedLabel)
    if (predictedLabel == 0):
        print("属于非低俗类")
    else:
        print("属于低俗类")
    
if __name__ == '__main__':
	testingNB()
