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
from sklearn import tree
import graphviz

def test_DT():
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

    ## Step 3: init NB
    print("Step 3: init NB...")
    #初始化贝叶斯分类器
    clf = tree.DecisionTreeClassifier(criterion='entropy')

    ## Step 4: training...
    print("Step 4: training...")
    #训练数据
    clf.fit(X, Y)
    
    ## Step 5: picture..
    print("Step 5: picture...")
    """
    dot_data = tree.export_graphviz(clf, out_file=None) 

    """
    #高级配置
    dot_data = tree.export_graphviz(clf, out_file=None, 
                            feature_names=iris.feature_names,  
                            class_names=iris.target_names,  
                            filled=True, rounded=True,  
                            special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render("tree")

if __name__ == '__main__':
    test_DT()