"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-09-16
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## Step 1: load data
print("Step 1: load data...")
#导入数据
iris = datasets.load_iris()

## Step 2: split data
print("Step 2: split data...")
#分离数据
# X = features
X = iris.data
# Y = label
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.6)

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
