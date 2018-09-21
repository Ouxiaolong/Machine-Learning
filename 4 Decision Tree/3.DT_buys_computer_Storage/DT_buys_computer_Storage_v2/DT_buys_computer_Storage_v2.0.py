"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-09-16
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding: utf-8 -*-
import pickle

"""
函数说明:读取决策树

Parameters:
    filename - 决策树的存储文件名
Returns:
    pickle.load(fr) - 决策树字典
"""
def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)
        
        
#测试
if __name__ == '__main__':
    myTree = grabTree('classifierStorage.txt')
    print(myTree)  
    
