"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-09-26
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
from sklearn import linear_model

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

"""
函数说明:读取数据

Parameters:
    filename - 文件名
Returns:
    xArr - x数据集
    yArr - y数据集
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
            if idx == 0:
                numFeature = len(eles)
            # 将数据转换成float型
            eles = list(map(float, eles)) 
            
            # 除最后一列都是feature，append(list)
            X.append(eles[:-1])   
            # 最后一列是实际值,同上
            Y.append([eles[-1]])    
     
    # 将X,Y列表转化成矩阵
    xArr = np.array(X)
    yArr = np.array(Y)
    
    return xArr,yArr

"""
函数说明:绘制数据集

Parameters:
    无
Returns:
    无
"""
def plotDataSet(xArr, yArr):
    n = len(xArr)	#数据个数
    xcord = []; ycord = []		#样本点
    
    for i in range(n):													
        xcord.append(xArr[i][1]); ycord.append(yArr[i])	#样本点

    fig = plt.figure()
    ax = fig.add_subplot(111)	#添加subplot
    #绘制样本点
    ax.scatter(xcord, ycord, s = 20, c = 'blue',alpha = .5)	
			
    plt.title('DataSet')		#绘制title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

"""
函数说明:定义模型函数

Parameters:
    theta, X
Returns:
    返回预测后的结果
"""
def predict(theta, X):
    return np.dot(X, theta)  # 此时的X为处理后的X

"""
函数说明:绘制回归曲线和数据点
Parameters
    X, Y, theta
Returns:
    无
"""
def plotRegression(X, Y, theta):
    
    xMat = np.mat(X)    #创建xMat矩阵
    yMat = np.mat(Y)    #创建yMat矩阵
    xCopy = xMat.copy()          #深拷贝xMat矩阵
    xCopy.sort(0)      #排序
    yHat = xCopy * theta      #计算对应的y值
    
    fig = plt.figure()
    ax = fig.add_subplot(111)    #添加subplot
    ax.plot(xCopy[:, 1], yHat, c = 'red')   #绘制回归曲线
    ax.scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue',alpha = .5) #绘制样本点
    
    plt.title('DataSet')     #绘制title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

"""
函数说明:绘制回归曲线和数据点
Parameters
    X, Y, theta
Returns:
    无
"""
def plotRegression_new(X, Y, theta):
    
    # 以下为训练集的预测值
    XCopy = X.copy()
    XCopy.sort(0)  # axis=0 表示列内排序
    yHat = predict(theta, XCopy)

    #print(XCopy[:,1].shape, yHat.shape, theta.shape)
    
    # 绘制回归直线
    plt.title('DataSet')		#绘制title
    plt.xlabel(u'x')
    plt.ylabel(u'y')
    #绘制回归线
    plt.plot(XCopy[:,1], yHat,color='r')
    #绘制样本点
    plt.scatter(X[:,1].flatten(), Y.T.flatten())
    plt.show()

"""
函数说明:计算相关系数
Parameters
    X, Y
Returns:
    相关系数
"""
def computeCorrelation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0 , len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX +=  diffXXBar**2
        varY += diffYYBar**2
    
    SST = math.sqrt(varX * varY)
    return SSR / SST

#测试
if __name__ == '__main__':
    ## Step 1: load data...
    print("Step 1: load data...")
    X, Y = loadDataSet('data.txt')
    #print(X.shape)
    #print(Y.shape)
  
    ## Step 2: show plot...
    print("Step 2: show plot...")
    plotDataSet(X, Y)
    
    ## Step 3: Create linear regression object...
    print("Step 3: Create linear regression object...")
    #regr = linear_model.LinearRegression()
    
    #岭回归
    regr = linear_model.Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
    
    ## Step 4: training...
    print("Step 4: training...")
    regr.fit(X, Y)
    
    ## Step 4: testing...
    print("Step 4: testing...")
    X_predict = [1, 18.945]
    # 到此，参数学习出来了，模型也就定下来了，若要预测新的实例，进行以下即可
    Y_predict = regr.predict([X_predict])
    
    ## Step 5: show the result...
    print("Step 5: show the result...")
    #定义向量 w = (w_1,..., w_p) 作为 coef_ ，定义 w_0 作为 intercept_ 。
    print("coef_:"+str(regr.coef_))
    print("intercept_:"+str(regr.intercept_)) 
    print("Y_predict:"+str(Y_predict))
    
    w0 = np.array(regr.intercept_)
    w1 = np.array(regr.coef_.T[1:2])
    
    #合并为一个矩阵
    theta = np.array([w0,w1])
    print(theta)
    
    ## Step 6: show Regression plot...
    print("Step 6: show Regression plot...")
    #plotRegression(X, Y, theta)
    plotRegression_new(X, Y, theta)
    
    ## Step 7: math Pearson...
    print("Step 7: math Pearson...")
    xMat = np.mat(X)                                                    #创建xMat矩阵
    yMat = np.mat(Y)                                                 #创建yMat矩阵
    yHat = predict(theta, xMat)
    
    Pearson = computeCorrelation(yHat, yMat)
    print(Pearson)
   