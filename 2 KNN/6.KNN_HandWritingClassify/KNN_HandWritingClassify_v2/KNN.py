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
from sklearn import neighbors
import os
from sklearn.metrics import accuracy_score

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

# load dataSet
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
    
    #构建kNN分类器
    #knn = kNN(n_neighbors = 3, algorithm = 'auto')
    knn = neighbors.KNeighborsClassifier(n_neighbors = 3)

    #拟合模型, train_x为训练矩阵,train_y为对应的标签
    knn.fit(train_x, train_y)
    
    #预测数据
    predict = knn.predict(test_x)
    
    for i in range(numTestSamples):
        
        print("Really Lable: %d \t KNN Lable :%d" % (test_y[i],predict[i]))
        if predict[i] == test_y[i]:
            matchCount += 1.0
    accuracy = float(matchCount) / numTestSamples
    ## step 4: show the result
    print("step 4: show the result...")  
    print("总共错了%d个数据\n" % (numTestSamples-matchCount))
    
    # 获得预测准确率
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    #方法一
    # print(accuracy_score(test_y, predict))
    
    #方法二
    print('准确率是: %.2f%%' % (accuracy * 100))

if __name__ == '__main__':
	testHandWritingClass()
