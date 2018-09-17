"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
"""
# @Date     : 2018-09-08
# @Author   : BruceOu
# @Language : Python3.6
"""
from sklearn import neighbors
from sklearn import datasets# 引入datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#第一步：读取数据集
iris = datasets.load_iris()# 获取所需数据集

#load_iris返回的结果有如下属性：
#feature_names - 分别为：sepal length (cm)， sepal width (cm)， petal length (cm)和 petal width (cm)
#data - 每行的数据，一共四列，每一列映射为feature_names中对应的值
#target - 每行数据对应的分类结果值（也就是每行数据的label值），其值为[0,1,2]
#target_names - target的值对应的名称，其值为['setosa' 'versicolor' 'virginica']

# save data
# f = open("iris.data.csv", 'wb')
# f.write(str(iris))
# f.close()

print(iris)

#第二步：分离数据
# X = features
X = iris.data
# Y = label
Y = iris.target
"""
那如何来使用数据呢？因为只有150行数据，所以为了验证算法的正确性，需要将数据分成两部分：
训练数据和测试数据，很幸运的是scikit-learn也提供了方便分离数据的方法train_test_split，我将数据分离成60%（即90条数据）用于训练，40%（即60条数据）用于测试
"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.6)

#第三步KNN分类
#初始化分类器
knn = neighbors.KNeighborsClassifier()

#训练
knn.fit(X_train, Y_train)

#第四步：预测数据
#predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])
predictedLabel = knn.predict(X_test)

# 获得预测准确率
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
print(accuracy_score(Y_test, predictedLabel))
print("predictedLabel is :")
print(predictedLabel)
