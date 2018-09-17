"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-09-08
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding: utf-8 -*-

import classify
import createData

if __name__ == '__main__':
    #创建数据集
    group, labels = createData.createDataSet()
    #测试集
    test = [18,90]
    #kNN分类
    #test_class = classify.classification(test, group, labels, 3)
    test_class = classify.classification_new(test, group, labels, 3)
    #打印分类结果
    print(test_class)
