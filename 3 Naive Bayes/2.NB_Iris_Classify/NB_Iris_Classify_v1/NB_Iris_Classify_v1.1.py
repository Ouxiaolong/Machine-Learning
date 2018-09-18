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
import csv#用于处理csv文件
import random#用于随机数

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
        if dataSet[x][-1] == 'Iris-setosa':
            data_Y.append(1)
        elif dataSet[x][-1] == 'Iris-versicolor':
            data_Y.append(2)
        elif dataSet[x][-1] == 'Iris-virginica':
            data_Y.append(3)
    return data_X, data_Y

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
            pass
            #print("the word: %s is not in my Vocabulary!" % word)
    #print(returnVec)
    return returnVec		#返回向量											


"""
函数说明:朴素贝叶斯分类器训练函数

Parameters:
	trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
	trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
	p1Vect,p2Vect,p3Vect,pAbusive,pBbusive,pCbusive

"""
def trainNB(trainMatrix,trainCategory):

    #计算训练的文档条数
    numTrainDocs = len(trainMatrix)							
    #print("numTrainDocs:" + str(numTrainDocs))
    
    #计算每篇文档的词条数
    numWords = len(trainMatrix[0])							
    #print("numWords:" + str(numWords))
    
    count = np.full(3, 0.0)
    
    for i in range(len(trainCategory)):
        if trainCategory[i] == 1:
            count[0] += 1
        elif trainCategory[i] == 2:
            count[1] += 1
        else:
            count[2] += 1
    
    pbusive = []
    #计算先验概率   
    for i in range(3):
        pb = count[i] /float(numTrainDocs)
        pbusive.append(pb)
    #print(pbusive)
 
    #创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    pNum = np.ones((3,numWords))
    #print(pNum)
    
    #分母初始化为0.0#避免其中一项为0的影响 
    pDenom =  np.full(3, 2.0)
    #print(pDenom)

    for i in range(numTrainDocs):
        #统计属于低俗类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
        if trainCategory[i] == 1:							
            pNum[0] += trainMatrix[i]
            pDenom[0] += sum(trainMatrix[i])
        elif trainCategory[i] == 2:							
            pNum[1] += trainMatrix[i]
            pDenom[1] += sum(trainMatrix[i])
        else:							
            pNum[2] += trainMatrix[i]
            pDenom[2] += sum(trainMatrix[i])
    
    pVect = []
    #避免下溢出问题
    for i in range(3):
        pV = np.log(pNum[i]/pDenom[i])								#相除        
        pVect.append(pV)
        
    return pVect, pbusive		#返回条件概率数组

"""
函数说明:朴素贝叶斯分类器分类函数

Parameters:
    vec2Classify - 待分类的词条数组
    pVec
    pClass
    lables - 标签
Returns:
    最大概率的标签
"""
def classifyNB(vec2Classify, pVec, pClass,lables):
    
    #概率列表
    p = []
    
    #从左到右对一个序列的项累计地应用有两个参数的函数，以此合并序列到一个单一值
    for i in range(len(lables)):
        result = sum(vec2Classify * pVec[i]) + np.log(pClass[i])
        p.append(result)
    
    #返回p中元素从小到大排序后的索引值
    # 按照升序进行快速排序，返回的是原数组的下标。
    # 比如，x = [30, 10, 20, 40]
    # 升序排序后应该是[10,20,30,40],他们的原下标是[1,2,0,3]
    # 那么，numpy.argsort(x) = [1, 2, 0, 3]
    sortedpIndices = np.argsort(p)
  
    #返回最大概率标签
    return lables[sortedpIndices[-1]]


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
    return (correct/float(len(testSet)))*100.0


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
    #prepare data
    trainSet = []#训练数据集
    testSet = []#测试数据集
    split = 0.8#分割的比例
    
    #lables = ['Iris-setosa','Iris-versicolor','Iris-virginica']
    
    lables = [1, 2, 3]
    
    loadDataset('C:/TensorFlow/irisdata.txt', split, trainSet, testSet)
    
    #数据集分割
    train_X,train_Y = segmentation_Data(trainSet)
    test_X,test_Y = segmentation_Data(testSet)
    
    print('Train set: ' + repr(len(trainSet)))
    print('Test set: ' + repr(len(testSet)))
    
    #print(train_X)
    #print(train_Y)

    #创建实验样本
    #listPosts,listClasses = loadDataSet()									
    #创建词汇表
    myVocabList = createVocabList(train_X)

    #向量化样本
    trainMat=[]
    for postinDoc in train_X: 
        #将实验样本向量化
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))				
    #print(trainMat)

    ## Step 2: training...
    print("Step 2: training...")
    #训练朴素贝叶斯分类器
    pV,pb = trainNB(np.array(trainMat),np.array(train_Y))	
    
    ## Step 3: testing
    print("Step 3: testing...")
    #测试样本
    #testEntry = [5.1,3.5,1.4,0.2]									
    #testEntry = [6.8,2.8,4.8,1.4]
    
    thisDoc = []
    predictedLabel = []
    
    for postinDoc in test_X: 
        #将实验样本向量化
        thisDoc.append(setOfWords2Vec(myVocabList, postinDoc))				
        
    #测试样本向量化
    #thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))				
   
    for i in range(len(thisDoc)):
        result = classifyNB(thisDoc[i],pV,pb,lables)
        predictedLabel.append(result)
    
    ## Step 4: show the result
    print("Step 4: show the result...")
    #print(predictedLabel)
    
    #print(test_Y)
    #准确率
    accuracy = getAccuracy(test_Y, predictedLabel)
    print('\nAccuracy: ' + repr(accuracy) + '%')
    
if __name__ == '__main__':
	testingNB()
