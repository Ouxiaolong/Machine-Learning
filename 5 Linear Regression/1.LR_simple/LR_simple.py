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

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

"""
函数说明:求解参数

Parameters:
    x,y
Returns:
    ws - 系数
"""
def fitSLR(x,y):
    #数据集长度
    n=len(x)
    dinominator = 0
    numerator=0
    
    for i in range(0,n):
        numerator += (x[i]-np.mean(x))*(y[i]-np.mean(y))
        dinominator += (x[i]-np.mean(x))**2
        
    #print("numerator:"+str(numerator))
    #print("dinominator:"+str(dinominator))
    
    b1 = numerator/float(dinominator)
    b0 = np.mean(y)/float(np.mean(x))
    
    ws = np.mat([[b0], [b1]])
    #返回系数
    return ws 


"""
函数说明:求解参数(矩阵的方式)
Parameters:
    xMat - x数据集
    yMat - y数据集
Returns:
    ws - 回归系数
"""
def MatfitSLR(xArr, yArr):
    #转化为MAT矩阵
    xMat = np.mat(xArr); yMat = np.mat(yArr)
        
    xTx = xMat.T * xMat     #根据文中推导的公示计算回归系数
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T*yMat)

    print(ws)
    
    return ws

"""
函数说明:预测数据

Parameters:
    x,b0,b1
Returns:
    预测值
"""
# y= b0+x*b1
def prefict(x, ws1):
    return ws[0] + x*ws[1]

"""
函数说明:绘制回归曲线和数据点
Parameters
    xArr, yArr,ws
Returns:
    无
"""
def plotRegression(xArr, yArr,ws):
    #创建xMat矩阵
    xMat = np.mat(xArr)                                                    
    #创建yMat矩阵
    yMat = np.mat(yArr)                                                    
    #深拷贝xMat矩阵
    xCopy = xMat.copy()                                                    
    
    xCopy.sort(0)  #排序
                
    #计算对应的y值                                       
    yHat = xCopy * ws                                                     
   
    fig = plt.figure()
    
    #添加subplot
    ax = fig.add_subplot(111)                                            
    
    #绘制回归曲线
    ax.plot(xCopy[:, 1], yHat, c = 'red')       
    
    ax.scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue',alpha = .5)               
    #绘制样本点
    plt.title('DataSet')     #绘制title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

#测试
if __name__ == '__main__':
    ## Step 1: load data...
    print("Step 1: load data...")
    x = [1,3,2,1,3]
    y = [14,24,18,17,27]
    
    xx = np.mat(x).T    
    m, n = xx.shape
    # 将第一列为1的矩阵，与原X相连X.shape
    X = np.concatenate((np.ones((m,1)), xx), axis=1)   
    #print(X) 
    
    #创建xMat矩阵
    xMat = np.mat(X)                                                    
    #创建yMat矩阵
    yMat = np.mat(y).T
    #print(xMat)
    #print(yMat)
    
    ## Step 2: training...
    print("Step 2: training...")
    ws = fitSLR(x, y)
    #矩阵求解参数
    #ws = MatfitSLR(xMat,yMat)
    
    ## Step 3: testing...
    print("Step 3: testing...")
    y_predict = prefict(6, ws)
    
    ## Step 4: show the plot...
    print("Step 4: show the plot...")
    plotRegression(X, y ,ws)
    
    ## Step 5: show the result...
    print("Step 5: show the result...")
    print("y_predict:"+str(y_predict))
       