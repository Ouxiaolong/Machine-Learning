"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-10-10
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding:utf-8 -*-
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    xx = np.arange(-3.0, 3.0, 0.1)
    
    theta0, theta1, theta2 = thetas
    
    yy = theta0 - (theta1*xx) / theta2
    plt.plot(xx, yy, color='red', linewidth  = 2)
    
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

"""
函数说明:绘制预测值与真实值

Parameters:
    X_test_std,Y_test,y_predict
Returns:
    无
"""
def plotfit(X_test_std,Y_test,y_predict):
    #画图对预测值和实际值进行比较
    p_x = range(len(X_test_std))
    plt.figure(figsize=(6,4),facecolor="w")
    plt.ylim(0,6)
    plt.plot(p_x,Y_test,"ro",markersize=8,zorder=3,label=u"真实值")
    plt.plot(p_x,y_predict,"go",markersize=14,zorder=2,label=u"预测值,$R^2$=%.3f" %lr.score(X_test,Y_test))
    plt.legend(loc="upper left")
    plt.xlabel(u"编号",fontsize=12)
    plt.ylabel(u"类型",fontsize=12)
    plt.title(u"Logistic算法对数据进行分类",fontsize=12)
    #plt.savefig("Logistic算法对数据进行分类.png")
    plt.show()
    

if __name__ == '__main__':
    ## Step 1: load data...
    print("Step 1: load data...")

    originX, Y = loadDataSet('data.txt')
    
    m,n = originX.shape

    X = np.concatenate((np.ones((m,1)), originX), axis=1)
    #print(X)
    #print(Y)
        
    ## Step 2: split data...
    print("Step 2: split data...")
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
    
    ## Step 3: standard data...
    print("Step 3: standard data...")
    #标准化
    sc=StandardScaler()
    sc.fit(X_train)#计算样本的均值和标准差
    X_train_std=sc.transform(X_train)
    X_test_std=sc.transform(X_test)

    start = time.time()
    ## Step 4: init LogisticRegression...
    print("Step 4: init LogisticRegression...")
    lr=LogisticRegression(C=1000.0,random_state=0)
 
    ## Step 5: training...
    print("Step 5: training ...")
    #输出y做出ravel()转换。不然会有警告
    lr.fit(X_train_std,Y_train.ravel())
    end = time.time()
    sk_time = end - start
    print('time:',sk_time)
    ## Step 6: show thetas...
    print("Step 6: show thetas...")
    #模型效果获取
    r = lr.score(X_train_std,Y_train)
    print("R值(准确率):",r)
       
    print("参数:",lr.coef_)
    print("截距:",lr.intercept_)
    
    w0 = np.array(lr.intercept_)
    w1 = np.array(lr.coef_.T[1:3])
    
    #合并为一个矩阵
    thetas = np.vstack((w0,w1))
    print('thetas:')
    print(thetas)
        
    ## Step 7: show sigmoid p...
    print("Step 7: show sigmoid p...")
    #sigmoid函数转化的值，即：概率p
    predict_p = lr.predict_proba(X_test_std)
    print(predict_p)     #sigmoid函数转化的值，即：概率pprint(y_predict)     #sigmoid函数转化的值，即：概率p
    
    ## Step 8: show boundary...
    print("Step 8: show boundary...")
    plotBoundary(X,Y,thetas)
    
    ## Step 9: predict ...
    print("Step 9: predict...")
    y_predict = lr.predict(X_test_std) 
    print(y_predict)
    
    ## Step 10: show plotfit ...
    print("Step 10: show plotfit...")
    #画图对预测值和实际值进行比较
    plotfit(X_test_std,Y_test,y_predict)
    
    
    