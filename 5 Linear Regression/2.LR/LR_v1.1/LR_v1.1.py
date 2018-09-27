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
from mpl_toolkits.mplot3d import axes3d
import math

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
函数说明:代价函数

Parameters:
    theta, X, Y
Returns:
    
"""
def J(theta, X, Y):
    '''定义代价函数'''
    m = len(X)
    return np.sum(np.dot((predict(theta,X)-Y).T, (predict(theta,X)-Y))/(2 * m))

"""
函数说明:梯度下降求解参数

Parameters:
    alpha, maxloop, epsilon, X, Y
Returns:
    theta, costs, thetas
    
"""
def bgd(alpha, maxloop, epsilon, X, Y):
    '''定义梯度下降公式，其中alpha为学习率控制步长，maxloop为最大迭代次数，epsilon为阈值控制迭代（判断收敛）'''
    m, n = X.shape # m为样本数，n为特征数，在这里为2

    # 初始化参数为零
    theta = np.zeros((2,1))
    
    count = 0 # 记录迭代次数
    converged = False # 是否收敛标志
    cost = np.inf # 初始化代价为无穷大
    costs = [] # 记录每一次迭代的代价值
    thetas = {0:[theta[0,0]], 1:[theta[1,0]]} # 记录每一轮theta的更新
    
    while count<= maxloop:
        if converged:
            break
        # 更新theta
        count = count + 1
        
        # 单独计算
        #theta0 = theta[0,0] - alpha / m * (predict(theta, X) - Y).sum()
        #theta1 = theta[1,0] - alpha / m * (np.dot(X[:,1][:,np.newaxis].T,(h(theta, X) - Y))).sum()   # 重点注意一下    
        # 同步更新
        #theta[0,0] = theta0
        #theta[1,0] = theta1
        #thetas[0].append(theta0)
        #thetas[1].append(theta1)
        
        # 一起计算
        theta = theta - alpha / (1.0 * m) * np.dot(X.T, (predict(theta, X)-Y))
        # X.T : n*m , h(theta, Y) : m*1 , np.dot(X.T, (predict(theta, X)- Y)) : n*1
        # 同步更新
        thetas[0].append(theta[0])
        thetas[1].append(theta[1])        
        
        # 更新当前cost
        cost = J(theta, X, Y)
        costs.append(cost)
        
        # 如果收敛，则不再迭代
        if cost<epsilon:
            converged = True
    return theta, costs, thetas 

"""
函数说明:最小二乘法求解（矩阵求解）
Parameters:
    xMat - x数据集
    yMat - y数据集
Returns:
    ws - 回归系数
"""
def standRegres(xArr, yArr):
    #转化为MAT矩阵
    xMat = np.mat(xArr); yMat = np.mat(yArr)
        
    xTx = xMat.T * xMat                            #根据文中推导的公示计算回归系数
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

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
函数说明:绘制代价曲线
Parameters
    X, Y, costs ,theta
Returns:
    无
"""
def plotloss(X, Y, costs, theta):
    
    # 以下为训练集的预测值
    XCopy = X.copy()
    XCopy.sort(0)  # axis=0 表示列内排序
    #print(XCopy[:,1].shape, yHat.shape, theta.shape)
    
    # 找到代价值的最大值、最小值，便于控制y轴范围
    print(np.array(costs).max(),np.array(costs).min())
    
    plt.xlim(-1,1600) # maxloop为1500
    plt.ylim(4,20)  
    plt.xlabel(u'迭代次数')
    plt.ylabel(u'代价函数J')
    plt.plot(range(len(costs)), costs)

"""
函数说明:绘制梯度下降过程
Parameters
    X, Y, theta
Returns:
    无
"""
def plotbgd(X, Y, theta):
    
    #查看theta0的范围
    print(np.array(thetas[0]).min(), np.array(thetas[0]).max()) 
    
    #查看theta1的范围
    print(np.array(thetas[1]).min(), np.array(thetas[1]).max())  
    
    # 准备网格数据，以备画梯度下降过程图
    # matplotlib
    size = 100
    theta0Vals = np.linspace(-10,10, size)
    theta1Vals = np.linspace(-2, 4, size)
    JVals = np.zeros((size, size))   # 按照theta0Vals与theta1Vals 将JVals初始化为0
    for i in range(size):
        for j in range(size):
            col = np.matrix([[theta0Vals[i]], [theta1Vals[j]]])
            JVals[i,j] = J(col, X, Y)
    
    theta0Vals, theta1Vals = np.meshgrid(theta0Vals, theta1Vals)
    JVals = JVals.T
    
    # 绘制3D代价函数图形
    contourSurf = plt.figure()
    ax = contourSurf.gca(projection='3d')
    
    ax.plot_surface(theta0Vals, theta1Vals, JVals,  rstride=2, cstride=2, alpha=0.3,
                    cmap=matplotlib.cm.rainbow, linewidth=0, antialiased=False)
    ax.plot(theta[0], theta[1], 'rx')
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel(r'$J(\theta)$')


"""
函数说明:绘制代价函数等高线图
Parameters
    X, Y
Returns:
    无
"""
def plotinline(X, Y):
    
    # 准备网格数据，以备画梯度下降过程图
    size = 100
    theta0Vals = np.linspace(-10,10, size)
    theta1Vals = np.linspace(-2, 4, size)
    JVals = np.zeros((size, size))   # 按照theta0Vals与theta1Vals 将JVals初始化为0
    for i in range(size):
        for j in range(size):
            col = np.matrix([[theta0Vals[i]], [theta1Vals[j]]])
            JVals[i,j] = J(col, X, Y)
    
    theta0Vals, theta1Vals = np.meshgrid(theta0Vals, theta1Vals)
    JVals = JVals.T
    
    # 绘制代价函数等高线图
    #matplotlib inline
    plt.figure(figsize=(12,6))
    CS = plt.contour(theta0Vals, theta1Vals, JVals, np.logspace(-2,3,30), alpha=.75)
    plt.clabel(CS, inline=1, fontsize=10)
    
    # 绘制最优解
    plt.plot(theta[0,0], theta[1,0], 'rx', markersize=10, linewidth=3)
    
    # 绘制梯度下降过程
    plt.plot(thetas[0], thetas[1], 'rx', markersize=3, linewidth=1) # 每一次theta取值
    plt.plot(thetas[0], thetas[1], 'r-',markersize=3, linewidth=1) # 用线连起来

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


"""
函数说明:绘制多条局部加权回归曲线
Parameters:
    无
Returns:
    无
"""
def plotlwlrRegression(xArr, yArr):

	yHat_1 = lwlrTest(xArr, xArr, yArr, 1.0)							#根据局部加权线性回归计算yHat
	yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)							#根据局部加权线性回归计算yHat
	yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)							#根据局部加权线性回归计算yHat
	xMat = np.mat(xArr)													#创建xMat矩阵
	yMat = np.mat(yArr)													#创建yMat矩阵
	srtInd = xMat[:, 1].argsort(0)										#排序，返回索引值
	xSort = xMat[srtInd][:,0,:]
	fig, axs = plt.subplots(nrows=3, ncols=1,sharex=False, sharey=False, figsize=(10,8))										

	axs[0].plot(xSort[:, 1], yHat_1[srtInd], c = 'red')						#绘制回归曲线
	axs[1].plot(xSort[:, 1], yHat_2[srtInd], c = 'red')						#绘制回归曲线
	axs[2].plot(xSort[:, 1], yHat_3[srtInd], c = 'red')						#绘制回归曲线
	axs[0].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = .5)				#绘制样本点
	axs[1].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = .5)				#绘制样本点
	axs[2].scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = .5)				#绘制样本点

	#设置标题,x轴label,y轴label
	axs0_title_text = axs[0].set_title(u'局部加权回归曲线,k=1.0')
	axs1_title_text = axs[1].set_title(u'局部加权回归曲线,k=0.01')
	axs2_title_text = axs[2].set_title(u'局部加权回归曲线,k=0.003')

	plt.setp(axs0_title_text, size=8, weight='bold', color='red')  
	plt.setp(axs1_title_text, size=8, weight='bold', color='red')  
	plt.setp(axs2_title_text, size=8, weight='bold', color='red')  

	plt.xlabel('X')
	plt.show()

"""
函数说明:使用局部加权线性回归计算回归系数w
Parameters:
    testPoint - 测试样本点
    xArr - x数据集
    yArr - y数据集
    k - 高斯核的k,自定义参数
Returns:
    ws - 回归系数
"""
def lwlr(testPoint, xArr, yArr, k = 1.0):

	xMat = np.mat(xArr); yMat = np.mat(yArr)
	m = np.shape(xMat)[0]
	weights = np.mat(np.eye((m)))	#创建权重对角矩阵
	for j in range(m):       #遍历数据集计算每个样本的权重
		diffMat = testPoint - xMat[j, :]     							
		weights[j, j] = np.exp(diffMat * diffMat.T/(-2.0 * k**2))
	xTx = xMat.T * (weights * xMat)										
	if np.linalg.det(xTx) == 0.0:
		print("矩阵为奇异矩阵,不能求逆")
		return
	ws = xTx.I * (xMat.T * (weights * yMat))	#计算回归系数
	return testPoint * ws

"""
函数说明:局部加权线性回归测试
Parameters:
    testArr - 测试数据集
    xArr - x数据集
    yArr - y数据集
    k - 高斯核的k,自定义参数
Returns:
    ws - 回归系数
"""
def lwlrTest(testArr, xArr, yArr, k=1.0):  
    m = np.shape(testArr)[0]		#计算测试数据集大小
    yHat = np.zeros(m)	
    
    for i in range(m):			#对每个样本点进行预测
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

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
    
    
    ## Step 3: training...
    print("Step 3: training...")
    alpha = 0.02 # 学习率
    maxloop = 1500 # 最大迭代次数
    epsilon = 0.01 # 收敛判断条件

    #方法一
    # 最优参数保存在theta中，costs保存每次迭代的代价值，thetas保存每次迭代更新的theta值
    theta, costs, thetas = bgd(alpha, maxloop, epsilon, X, Y) 
    #print(theta, costs[:5], thetas)
    
    #方法二
    #theta = standRegres(X, Y) 
    print(theta)

    ## Step 4: testing...
    print("Step 4: testing...")
    X_predict = [1, 18.945]
    # 到此，参数学习出来了，模型也就定下来了，若要预测新的实例，进行以下即可
    Y_predict = predict(theta, X_predict)
    
    ## Step 5: show the result...
    print("Step 5: show the result...")
    print(Y_predict)
    
    ## Step 6: show Regression plot...
    print("Step 6: show Regression plot...")
    #plotRegression(X, Y, theta)
    plotRegression_new(X, Y, theta)
    
    ## Step 7: show lossplot...
    print("Step 7: show lossplot...")
    plotloss(X, Y, costs, theta)
    
    ## Step 8: show plotbgd...
    print("Step 8: show plotbgd...")
    plotbgd(X, Y, theta)

    ## Step 9: show plotinline...
    print("Step 9: show plotinline...")
    plotinline(X, Y)
    
    ## Step 10: math Pearson...
    print("Step 10: math Pearson...")
    xMat = np.mat(X)                                                    #创建xMat矩阵
    yMat = np.mat(Y)                                                 #创建yMat矩阵
    yHat = predict(theta, xMat)
    
    Pearson = computeCorrelation(yHat, yMat)
    print(Pearson)

    ## Step 11: plotlwlrRegression...
    print("Step 11: plotlwlrRegression...")
    plotlwlrRegression(X, Y)
