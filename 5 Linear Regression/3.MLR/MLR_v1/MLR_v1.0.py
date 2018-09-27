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
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.ticker as mtick

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
    '''加载数据'''
    X = []
    Y = []
    with open(filename, 'rb') as f:
        for idx, line in enumerate(f):
            line = line.decode('utf-8').strip()
            if not line:
                continue
                
            eles = line.split()
            if idx == 0:
                numFea = len(eles)
            eles = list(map(float, eles))
            
            X.append(eles[:-1])
            Y.append([eles[-1]])
     
    # 将X,Y列表转化成矩阵
    xArr = np.array(X)
    yArr = np.array(Y)
    
    return xArr,yArr

"""
函数说明:特征标准化处理

Parameters:
    X - 样本集
Returns: 
    X, values - 标准后的样本集
"""
def standarize(X):
    m, n = X.shape
    values = {}  # 保存每一列的mean和std，便于对预测数据进行标准化
    for j in range(n):
        features = X[:,j]
        meanVal = features.mean(axis=0)
        stdVal = features.std(axis=0)
        values[j] = [meanVal, stdVal]
        if stdVal != 0:
            X[:,j] = (features - meanVal) / stdVal
        else:
            X[:,j] = 0
    return X, values

"""
函数说明:代价函数

Parameters:
    theta, X, Y
Returns:
    
"""
def J(theta, X, Y):
    m = len(X)
    return np.sum(np.dot((predict(theta, X) - Y).T, (predict(theta, X) - Y)) / (2 * m))

"""
函数说明:梯度下降求解参数

Parameters:
    alpha, X, Y, maxloop, epsilon
Returns:
    theta, costs, thetas
    
"""
def bgd(alpha, X, Y, maxloop, epsilon):

    m, n = X.shape
    theta = np.zeros((n,1))  #初始化参数为0
    
    count = 0 # 记录迭代次数
    converged = False # 是否已收敛的标志
    cost = np.inf # 初始化代价值为无穷大
    costs = [J(theta, X, Y),] # 记录每一次的代价值
    
    thetas = {}  # 记录每一次参数的更新
    for i in range(n):
        thetas[i] = [theta[i,0],]
        
    while count <= maxloop:
        if converged:
            break
        count += 1
        
        # n个参数计算，并存入thetas中（单独计算）
        #for j in range(n):
        #    deriv = np.sum(np.dot(X[:,j].T, (h(theta, X) - Y))) / m
        #    thetas[j].append(theta[j,0] - alpha*deriv)
        # n个参数在当前theta中更新  
        #for j in range(n):
        #    theta[j,0] = thetas[j][-1]
        
        #n个参数同时计算更新值
        theta = theta - alpha * 1.0 / m * np.dot(X.T, (predict(theta, X) - Y))
        #添加到thetas中
        for j in range(n):
            thetas[j].append(theta[j,0])
            
        # 记录当前参数的函数代价，并存入costs
        cost = J(theta, X, Y)
        costs.append(cost)
            
        if abs(costs[-1] - costs[-2]) < epsilon:
            converged = True
    
    return theta, thetas, costs

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
    
    # 打印拟合平面
    fittingFig = plt.figure(figsize=(16, 12))
    ##title = 'bgd: rate=%.3f, maxloop=%d, epsilon=%.3f \n'%(alpha,maxloop,epsilon)
    ax=fittingFig.gca(projection='3d')
    
    xx = np.linspace(0,200,25)
    yy = np.linspace(0,5,25)
    zz = np.zeros((25,25))
    for i in range(25):
        for j in range(25):
            normalizedSize = (xx[i]-values[0][0])/values[0][1]
            normalizedBr = (yy[j]-values[1][0])/values[1][1]
            x = np.matrix([[1,normalizedSize, normalizedBr]])
            zz[i,j] =predict(theta, x)
    xx, yy = np.meshgrid(xx,yy)
    ax.zaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cm.rainbow, alpha=0.1, antialiased=True)
    
    xs = X[:, 0].flatten()
    ys = X[:, 1].flatten()
    zs = Y[:, 0].flatten()
    ax.scatter(xs, ys, zs, c='b', marker='o')
    
    ax.set_xlabel(u'面积')
    ax.set_ylabel(u'卧室数')
    ax.set_zlabel(u'估价')

"""
函数说明:绘制代价函数曲线
Parameters
    costs
Returns:
    无
"""
def plotinline(costs):
    
    errorsFig = plt.figure()
    ax = errorsFig.add_subplot(111)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    
    ax.plot(range(len(costs)), costs)
    ax.set_xlabel(u'迭代次数')
    ax.set_ylabel(u'代价函数')

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
    originX, Y = loadDataSet('houses.txt')
    #print(originX.shape, Y.shape)
  
    # 对特征X增加x0列
    m,n = originX.shape
    X, values = standarize(originX.copy())
    X = np.concatenate((np.ones((m,1)), X), axis=1)
    #print(X.shape, Y.shape)
    #print(X[:3],values)
    
    ## Step 2: training...
    print("Step 2: training...")
    alpha = 1 # 学习率
    maxloop = 5000 # 最大迭代次数
    epsilon = 0.000001

    # 最优参数保存在theta中，costs保存每次迭代的代价值，thetas保存每次迭代更新的theta值
    theta , thetas, costs= bgd(alpha, X, Y, maxloop, epsilon)
    #print(theta , thetas, costs)
    
    print(theta)

    ## Step 4: testing...
    print("Step 4: testing...")
    normalizedSize = (70-values[0][0])/values[0][1]
    normalizedBr = (2-values[1][0])/values[1][1]
    predicateX = np.matrix([[1, normalizedSize, normalizedBr]])
    
    # 到此，参数学习出来了，模型也就定下来了，若要预测新的实例，进行以下即可
    price = predict(theta, predicateX)
    
    ## Step 5: show the result...
    print("Step 5: show the result...")
    print('70㎡两居估价: ￥%.4f万元'%price)
    
    ## Step 6: show Regression plot...
    print("Step 6: show Regression plot...")
    plotRegression(X, Y, theta)

    ## Step 7: show plotinline...
    print("Step 7: show plotinline...")
    plotinline(costs)
    
    ## Step 8: math Pearson...
    print("Step 8: math Pearson...")
    xMat = np.mat(X)      #创建xMat矩阵
    yMat = np.mat(Y)      #创建yMat矩阵
    yHat = predict(theta, xMat)
    
    Pearson = computeCorrelation(yHat, yMat)
    print(Pearson)