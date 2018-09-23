"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-09-16
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding: utf-8 -*-
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing

#测试
if __name__ == '__main__':
    
    ## Step 1: load data
    print("Step 1: load data...")

    # Read in the csv file and put features into list of dict and list of class label
    Data = open("C:/TensorFlow/data.csv", "rt")
    
    #读取文件的原始数据
    reader = csv.reader(Data)#返回的值是csv文件中每行的列表，将每行读取的值作为列表返回
    
    #3.x版本使用该语法，2.7版本则使用headers=reader.next()
    headers = next(reader)#读取行的文件对象,reader指向下一行
    #headers存放的是csv的第一行元素，也是后文rowDict的键值
    #print("headers :\n" + str(headers))
    
    featureList = []
    labelList = []
    
    for row in reader:
        labelList.append(row[len(row)-1])
        rowDict = {}
        for i in range(1, len(row)-1):
            rowDict[headers[i]] = row[i]
        featureList.append(rowDict)
    
    #print("featureList:\n" + str(featureList))
    #print("labelList:\n" + str(labelList))
    
    ## Step 2: Vetorize data...
    print("Step 2: Vetorize data...")

    #提取数据
    # Vetorize features
    vec = DictVectorizer()#初始化字典特征抽取器
    dummyX = vec.fit_transform(featureList).toarray()
    # 查看提取后的特征值
    #输出转化后的特征矩阵
    #print("dummyX: \n" + str(dummyX))
    #输出各个维度的特征含义
    #print(vec.get_feature_names())

    # vectorize class labels
    lb = preprocessing.LabelBinarizer()# 将标签矩阵二值化
    dummyY = lb.fit_transform(labelList)
    #print("dummyY: \n" + str(dummyY))
    
    ## Step 3: init DT...
    print("Step 3: init DT...")
    # Using decision tree for classification
    # http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    # clf = tree.DecisionTreeClassifier()
    ## criterion可选‘gini’, ‘entropy’，默认为gini(对应CART算法)，entropy为信息增益（对应ID3算法）
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    
    ## Step 4: training...
    print("Step 4: training...")
    clf = clf.fit(dummyX, dummyY)

   #预测数据
    oneRowX = dummyX[0, :]
    #print("oneRowX: " + str(oneRowX))
    
    newRowX = oneRowX
    newRowX[0] = 1
    newRowX[2] = 0
    print("newRowX: " + str(newRowX))
    
    ## Step 5: testing
    print("Step 5: testing...")
    #predictedLabel = clf.predict([newRowX])#方法一
    predictedLabel = clf.predict(newRowX.reshape(1,-1))#方法二

    ## Step 6: show the result
    print("Step 4: show the result...")
    #print("predictedLabel" + str(predictedLabel))

    if predictedLabel == 1:
        print("要购买")
    else:
        print("不购买")
