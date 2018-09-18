"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-09-16
# @Author   : BruceOu
# @Language : Python3.6
"""
# -*- coding: utf-8 -*-

import pandas as pd
import time

from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    ## Step 1: load data
    print("Step 1: load data...")
    time_1 = time.time()

    raw_data = pd.read_csv('train.csv', header=0)  # 读取csv数据
    data = raw_data.values

    features = data[::, 1::]
    labels = data[::, 0]
    
    # 随机选取33%数据作为测试集，剩余为训练集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=0)

    time_2 = time.time()
    print('read data cost %f seconds' % (time_2 - time_1))


    ## Step 2: training...
    print("Step 2: training...")
    clf = MultinomialNB(alpha=1.0) # 加入laplace平滑
    clf.fit(train_features, train_labels)
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))

    ## Step 3: testing
    print("Step 3: testing...")
    test_predict = clf.predict(test_features)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))

    ## Step 4: show the result
    print("Step 4: show the result...")
    score = accuracy_score(test_labels, test_predict)
    print("The accruacy score is %f" % score)

