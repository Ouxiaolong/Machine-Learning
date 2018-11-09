"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-11-07
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False


"""
函数说明:读取数据

Parameters:
    filename - 文件名
Returns:
    dataMat - data数据集
    labelMat - 标签数据集
"""
def loadData(filename):
    
    data = np.loadtxt(filename)
    
    dataMat = data[:, 0:2]
    labelMat = data[:, 2]
    
    return dataMat, labelMat

"""
函数说明:读取数据

Parameters:
    filename - 文件名
Returns:
    xArr - data数据集
    yArr - 标签数据集
"""
def loadDataSet(filename):
    X = []
    Y = []
    
    with open(filename, 'rb') as f:
        for idx, line in enumerate(f):
            line = line.decode('utf-8').strip()
            if not line:
                continue
            eles = line.split()
            eles = list(map(float, eles))
            
            if idx == 0:
                numFea = len(eles)
            #去掉每行的最后一个数据再放入X中
            X.append(eles[:-1])
            #获取每行的租后一个数据
            Y.append([eles[-1]])
     
    # 将X,Y列表转化成矩阵
    xArr = np.array(X)
    yArr = np.array(Y)
    
    return xArr,yArr


"""
函数说明:训练数据集

Parameters:
    dataMat - 数据集
    labelMat - 标签集
    alpha
    b - bias
    eta - dataMat
Returns:
    weight, bias
"""
def trainPerceptron(dataMat, labelMat, alpha, b, eta):

    m =len(dataMat)
    flag = True
    while flag:
        for i in range(m):
            if (labelMat[i, 0] * (np.sum((alpha * labelMat * np.dot(dataMat, dataMat[i].T).reshape((m, 1)))) + b)) <= 0:
                alpha[i] = alpha[i] + eta
                b = b + eta * labelMat[i]
                flag = True
                break
            else:
                flag = False
    w = np.dot(dataMat.T, alpha * labelMat)
    return w, b
 
"""
函数说明:可视化展示分类结果

Parameters:
    dataMat - 数据集
    labelMat - 标签集
    weight
    bias
Returns:
    无
"""
def plotResult(dataMat, labelMat, weight, bias):
    fig = plt.figure()
    axes = fig.add_subplot(111)

    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    for i in range(len(labelMat)):
        if (labelMat[i] == -1):
            type1_x.append(dataMat[i][0])
            type1_y.append(dataMat[i][1])

        if (labelMat[i] == 1):
            type2_x.append(dataMat[i][0])
            type2_y.append(dataMat[i][1])

    #方法一
    #axes.scatter(type1_x, type1_y, marker='x', s=20, c='red')
    #axes.scatter(type2_x, type2_y, marker='o', s=20, c='blue')
    #方法二
    plt.plot(type1_x, type1_y, 'bo', type2_x, type2_y, 'rx')
    
    y = (0.1 * -weight[0] / weight[1] + -bias / weight[1], 4.0 * -weight[0] / weight[1] + -bias / weight[1])
    axes.add_line(Line2D((0.1, 4.0), y, linewidth=3, color='yellow'))
    plt.axis([0, 5, 0, 5])
    plt.grid(True)
    plt.title('Perceptron Algorithm ')
    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.show()
   
if __name__ == "__main__":
    ## Step 1: load data...
    print("Step 1: load data...")
    dataMat, labelMat = loadDataSet('testSet.txt')

    ## Step 2: training...
    print("Step 2: training...")
    alpha = np.zeros((len(dataMat), 1))
    b = 0
    eta = 1
    weight, bias = trainPerceptron(dataMat, labelMat, alpha, b, eta)
   
    ## Step 3: show the result...
    print("Step 3: show the result...")
    print("weight,bias:",weight,bias)
    
    ## Step 4: show the picture...
    print("Step 4: show the picture...")
    plotResult(dataMat, labelMat, weight, bias)


