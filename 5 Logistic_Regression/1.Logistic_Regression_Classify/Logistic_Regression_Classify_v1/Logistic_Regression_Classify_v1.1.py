"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-09-29
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding:utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

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
函数说明:定义sigmoid函数

Parameters:
    z
Returns:
    返回
"""
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

"""
函数说明:定义代价函数

Parameters:
    theta, X, Y, theLambda
Returns:
    返回
"""
def J(theta, X, Y, theLambda=0):
    m, n = X.shape
    h = sigmoid(np.dot(X, theta))
    J = (-1.0/m)*(np.dot(np.log(h).T,Y)+np.dot(np.log(1-h).T,1-Y)) + (theLambda/(2.0*m))*np.sum(np.square(theta[1:]))
    if np.isnan(J[0]):
        return np.inf
    # 其实J里面只有一个数值，需要取出该数值
    return J.flatten()[0]

"""
函数说明:梯度上升求解参数

Parameters:
    X, Y, options
Returns:
    返回参数
"""
def gradient(X, Y, options):
    '''
    options.alpha 学习率
    options.theLambda 正则参数λ
    options.maxLoop 最大迭代轮次
    options.epsilon 判断收敛的阈值
    options.method
        - 'sgd' 随机梯度上升
        - 'bgd' 批量梯度上升
    '''
    
    m, n = X.shape
    
    # 初始化模型参数，n个特征对应n个参数
    theta = np.zeros((n,1))
    
    cost = J(theta, X, Y)  # 当前误差
    costs = [cost]
    thetas = [theta]
    
    # Python 字典dict.get(key, default=None)函数返回指定键的值，如果值不在字典中返回默认值
    alpha = options.get('alpha', 0.01)
    epsilon = options.get('epsilon', 0.00001)
    maxloop = options.get('maxloop', 1000)
    theLambda = float(options.get('theLambda', 0)) # 后面有 theLambda/m 的计算，如果这里不转成float，后面这个就全是0
    method = options.get('method', 'bgd')
    
    # 定义随机梯度上升
    def _sgd(theta):
        count = 0
        converged = False
        while count < maxloop:
            if converged :
                break
            # 随机梯度上升，每一个样本都更新
            for i in range(m):
                h =sigmoid(np.dot(X[i].reshape((1,n)), theta))
                
                theta = theta + alpha*((1.0/m)*X[i].reshape((n,1))*(Y[i]-h) + (theLambda/m)*np.r_[[[0]], theta[1:]])
                
                thetas.append(theta)
                cost = J(theta, X, Y, theLambda)
                costs.append(cost)
                if abs(costs[-1] - costs[-2]) < epsilon:
                    converged = True
                    break
            count += 1
        return thetas, costs, count
    
    # 定义批量梯度上升
    def _bgd(theta):
        count = 0
        converged = False
        while count < maxloop:
            if converged :
                break
            
            h = sigmoid(np.dot(X, theta))
            theta = theta + alpha*((1.0/m)*np.dot(X.T, (Y-h)) + (theLambda/m)*np.r_[[[0]],theta[1:]])
            
            thetas.append(theta)
            cost = J(theta, X, Y, theLambda)
            costs.append(cost)
            if abs(costs[-1] - costs[-2]) < epsilon:
                converged = True
                break
            count += 1
        return thetas, costs, count
    
    methods = {'sgd': _sgd, 'bgd': _bgd}
    return methods[method](theta)  

"""
函数说明:绘制决策边界

Parameters:
    X, thetas
Returns:
    无
"""
def plotBoundary(X,Y,thetas):
        
    # 绘制决策边界
    plt.figure(figsize=(6,4))
    for i in range(len(X)):
        x = X[i]
        if Y[i] == 1:
            plt.scatter(x[1], x[2], marker='*', color='blue', s=50)
        else:
            plt.scatter(x[1], x[2], marker='o', color='green', s=50)
    
    hSpots = np.linspace(X[:,1].min(), X[:,1].max(), 100)
    theta0, theta1, theta2 = thetas[-1]
    
    vSpots = -(theta0+theta1*hSpots)/theta2
    plt.plot(hSpots, vSpots, color='red', linewidth=.5)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

"""
函数说明:绘制代价函数图像

Parameters:
    costs
Returns:
    返回参数
"""
def plotCost(costs):
    # 绘制误差曲线
    plt.figure(figsize=(8,4))
    plt.plot(range(len(costs)), costs)
    plt.xlabel(u'迭代次数')
    plt.ylabel(u'代价J')    

"""
函数说明:绘制参数曲线

Parameters:
    thetass - 参数
Returns:
    返回参数
"""
def plotthetas(thetas):
    # 绘制参数theta变化
    thetasFig, ax = plt.subplots(len(thetas[0]))
    thetas = np.asarray(thetas)
    for idx, sp in enumerate(ax):
        thetaList = thetas[:, idx]
        sp.plot(range(len(thetaList)), thetaList)
        sp.set_xlabel('Number of iteration')
        sp.set_ylabel(r'$\theta_%d$'%idx)

if __name__ == '__main__':
    ## Step 1: load data...
    print("Step 1: load data...")

    originX, Y = loadDataSet('data.txt')
    
    m,n = originX.shape

    X = np.concatenate((np.ones((m,1)), originX), axis=1)
    #print(X)
    #print(Y)
        
    ## Step 2: training bgd...
    print("Step 2: training bgd...")
    # bgd批量梯度上升
    # 设置超参数
    options = {
        'alpha':0.05,
        'epsilon':0.00000001,
        'maxloop':100000,
        'method':'bgd' # sgd
    }
    
    # 训练模型
    start = time.time()
    bgd_thetas, bgd_costs, bgd_iterationCount = gradient(X, Y, options)
    print("bgd_thetas:")
    print(bgd_thetas[-1])
    end = time.time()
    bgd_time = end - start
    print("bgd_time:"+str(bgd_time))
    
    ## Step 3: show boundary bgd...
    print("Step 3: show boundary bgd...")
    plotBoundary(X,Y,bgd_thetas)

    ## Step 4: show costs bgd...
    print("Step 4: show costs bgd...")
    plotCost(bgd_costs)

    ## Step 5: show thetas bgd...
    print("Step 5: show thetas bgd...")
    plotthetas(bgd_thetas)
   
    print("======================================================")
    ## Step 2: training sgd...
    print("Step 2: training sgd...")
    
    # sgd随机梯度上升
    options = {
        'alpha':0.5,
        'epsilon':0.00000001,
        'maxloop':100000,
        'method':'sgd' 
    }
    
    # 训练模型
    start = time.time()
    sgd_thetas, sgd_costs, sgd_iterationCount = gradient(X, Y, options)
    print("sgd_thetas:")
    print(sgd_thetas[-1])
    end = time.time()
    sgd_time = end - start
    print("sgd_time:"+str(sgd_time))
    
    ## Step 3: show boundary sgd...
    print("Step 3: show boundary sgd...")
    plotBoundary(X,Y,sgd_thetas)

    ## Step 4: show costs sgd...
    print("Step 4: show costs sgd...")
    plotCost(sgd_costs)

    ## Step 5: show thetas sgd...
    print("Step 5: show thetas sgd...")
    plotthetas(sgd_thetas)