"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-09-16
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding: utf-8 -*-
import random
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

Parameters:
	dataSet - 整理的样本数据集
Returns:
	vocabSet - 返回不重复的词条列表，也就是词汇表
"""
def createVocabList(dataSet):
    vocabSet = set([])  					#创建一个空的不重复列表
    for document in dataSet:				
        vocabSet = vocabSet | set(document) #取并集
    return list(vocabSet)

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
        if word in vocabList:											
            #如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else: 
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec													#返回文档向量

"""
函数说明:接收一个大字符串并将其解析为字符串列表

Parameters:
    无
Returns:
    无
"""
def textParse(bigString):    
    #将特殊符号作为切分标志进行字符串切分，即非字母、非数字                                               #将字符串转换为字符列表
    listOfTokens = re.split(r'\W*', bigString)  
    
    #print(listOfTokens)
    
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]            #除了单个字母，例如大写的I，其它单词变成小写

"""
函数说明:测试朴素贝叶斯分类器

Parameters:
    无
Returns:
    无
"""
def spamTest():
    ## Step 1: load data
    print("Step 1: load data...")

    docList = []
    classList = []
    fullText = []
    
    for i in range(1, 26):                                                  #遍历25个txt文件
        wordList = textParse(open('email/spam/%d.txt' % i, 'rt').read())     #读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)                                                 #标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('email/ham/%d.txt' % i, 'rt').read())      #读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)                                                 #标记非垃圾邮件，1表示垃圾文件    
    
    vocabList = createVocabList(docList)                                    #创建词汇表，不重复
    trainingSet = list(range(50))
                                
    #创建存储训练集的索引值的列表和测试集的索引值的列表                        
    testSet = []
    
    for i in range(10):                                                     #从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0, len(trainingSet)))                #随机选取索索引值
        testSet.append(trainingSet[randIndex])                              #添加测试集的索引值
        del(trainingSet[randIndex])                                         #在训练集列表中删除添加到测试集的索引值
    
    #创建训练集矩阵和训练集类别标签系向量 
    trainMat = []
    trainClasses = []                                        
                 
    ## Step 2: training...
    print("Step 2: training...")

    #遍历训练集
    for docIndex in trainingSet:                                            
        #将生成的词集模型添加到训练矩阵中
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))       
        #将类别添加到训练集类别标签系向量中
        trainClasses.append(classList[docIndex])                            
    
    X_train, X_test, Y_train, Y_test = train_test_split(trainMat, trainClasses, test_size=.6)

    ## Step 3: init NB
    print("Step 3: init NB...")
    #初始化贝叶斯分类器
    gnb = GaussianNB()
    
    ## Step 4: training...
    print("Step 4: training...")
    #训练数据
    gnb.fit(X_train, Y_train)
    
    ## Step 5: testing
    print("Step 5: testing...")
    #预测数据
    predictedLabel =  gnb.predict(X_test)
    #predictedLabel = gnb.fit(X_train, Y_train).predict(X_test)

    ## Step 6: show the result
    print("Step 6: show the result...")
    #求准确率
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    print(accuracy_score(Y_test, predictedLabel))
    print("predictedLabel is :")
    print(predictedLabel)

if __name__ == '__main__':
    spamTest()
