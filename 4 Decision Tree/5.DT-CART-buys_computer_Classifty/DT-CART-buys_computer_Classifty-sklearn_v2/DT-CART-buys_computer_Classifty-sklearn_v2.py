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
import graphviz
from sklearn import tree

def createDataSet():
    #数据集
    dataSet = [[0, 0, 0, 2],
               [0, 0, 1, 2],
               [1, 0, 0, 2],
               [2, 0, 0, 1],
               [2, 1, 0, 0], 
               [2, 1, 1, 0],
               [1, 1, 1, 0], 
               [0, 0, 0, 1],
               [0, 1, 0, 0],
               [2, 1, 0, 1], 
               [0, 1, 1, 1], 
               [1, 0, 1, 1], 
               [1, 1, 0, 2],
               [2, 0, 2, 2]]
    #特征标签
    labels = [0,0,1,1,1,0,1,0,1,1,1,1,1,0]

    #返回数据集和分类属性
    return dataSet, labels 

#测试
if __name__ == '__main__':
    ## Step 1: load data
    print("Step 1: load data...")
    #方式一
    df=pd.read_csv('data.csv')
    data=df.values[:-1,1:5]
    dataSet=data[:]
    
    labels=df.values[:-1,5:6]
    
    #方式二
    #dataSet,labels = createDataSet()
    
    #print(dataSet)
    #print(labels)
    
    ## Step 2: init DT...
    print("Step 2: init DT...")

    # criterion可选‘gini’, ‘entropy’，默认为gini(对应CART算法)，entropy为信息增益（对应ID3算法）    
    #clf = tree.DecisionTreeClassifier(criterion = 'gini')
    
    clf = tree.DecisionTreeClassifier()
    ## Step 3: training...
    print("Step 3: training...")
    clf = clf.fit(dataSet, labels)
    
    ## Step 4: picture...
    print("Step 4: picture...")
    """
    dot_data = tree.export_graphviz(clf, out_file=None) 

    """
    #高级配置
    dot_data = tree.export_graphviz(clf, out_file=None, 
                            filled=True, rounded=True,  
                            special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render("tree")
    
    ## Step 5: testing
    print("Step 5: testing...")
    test = [1,0,0,2]
    predictedLabel = clf.predict([test])
    
    # Step 6: show the result
    print("Step 6: show the result...")
    print("predictedLabel" + str(predictedLabel))