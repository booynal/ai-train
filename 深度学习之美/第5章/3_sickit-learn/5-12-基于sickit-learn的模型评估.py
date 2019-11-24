import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 加载IRIS数据集
sickit_iris = datasets.load_iris()

# 转换为Pandas的DataFrame格式，以便于观察数据
pd_iris = pd.DataFrame(
    data=np.c_[sickit_iris['data'], sickit_iris['target']],
    columns=np.append(sickit_iris.feature_names, ['y'])
)
# print(pd_iris.head(3))

# 选择全部特征参与训练模型
X = pd_iris[sickit_iris.feature_names]
y = pd_iris['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print('X_train:', X_train.count())


# 1. 选择模型
knn = KNeighborsClassifier(n_neighbors=10)

# 2. 拟合模型（训练模型）
knn.fit(X_train, y_train)

# 3. 预测新数据，参数为2D数组
predict_on_train = knn.predict(X_train)
predict_on_test = knn.predict(X_test)
print('predict_on_train:', predict_on_train)
print('predict_on_test:', predict_on_test)

accuracy_train = metrics.accuracy_score(y_train, predict_on_train)
accuracy_test = metrics.accuracy_score(y_test, predict_on_test)
print('训练集预测准确率: {:.3%}'.format(accuracy_train))
print('测试集预测准确率: {:.3%}'.format(accuracy_test))
