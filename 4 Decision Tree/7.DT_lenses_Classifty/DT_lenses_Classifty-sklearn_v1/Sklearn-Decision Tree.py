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
import graphviz 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO
import pandas as pd

if __name__ == '__main__':
    ## Step 1: load data
    print("Step 1: load data...")

    with open('lenses.txt', 'r') as fr:	#加载文件
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]		#处理文件
	
    #提取每组数据的类别，保存在列表里
    lenses_target = []														
    
    for each in lenses:
        lenses_target.append(each[-1])
    #print(lenses_target)
    
    	
    ## Step 2: encoder...
    print("Step 2: encoder...")
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']#特征标签		
    
    lenses_list = []	#保存lenses数据的临时列表
    lenses_dict = {}		#保存lenses数据的字典，用于生成pandas
	
    for each_label in lensesLabels:	#提取信息，生成字典
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
	
    # print(lenses_dict)		#打印字典信息
    lenses_pd = pd.DataFrame(lenses_dict)	#生成pandas.DataFrame  
	
    #打印pandas.DataFrame
    # print(lenses_pd)														
    
    #创建LabelEncoder()对象，用于序列化		
    le = LabelEncoder()														
    for col in lenses_pd.columns:	#序列化
        lenses_pd[col] = le.fit_transform(lenses_pd[col])

    #print(lenses_pd)			#打印编码信息
    #print(lenses_pd.values.tolist())
	
    ## Step 3: init DT...
    print("Step 3: init DT...")

    clf = tree.DecisionTreeClassifier(max_depth = 4)						#创建DecisionTreeClassifier()类
	
    ## Step 4: training...
    print("Step 4: training...")
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)					#使用数据，构建决策树

	
    ## Step 5: testing...
    print("Step 5: testing...")
    print(clf.predict([[1,1,1,0]]))											#预测

	
    ## Step 6: picture...
    print("Step 6: picture...")
    #dot_data = tree.export_graphviz(clf, out_file=None) 

    #高级配置
    dot_data = tree.export_graphviz(clf, out_file=None,  
                            filled=True, rounded=True,  
                            special_characters=True)  
    
    
    graph = graphviz.Source(dot_data)  
    graph.render("tree")
