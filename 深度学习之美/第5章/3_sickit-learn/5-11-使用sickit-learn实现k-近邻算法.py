import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# 加载IRIS数据集
sickit_iris = datasets.load_iris()
print('sickit_iris.data:', sickit_iris.data)
print('sickit_iris.target:', sickit_iris.target)
print('sickit_iris.target_names:', sickit_iris.target_names)
print('sickit_iris.feature_names:', sickit_iris.feature_names)
print('sickit_iris.filename:', sickit_iris.filename)

# 转换为Pandas的DataFrame格式，以便于观察数据
pd_iris = pd.DataFrame(
    data=np.c_[sickit_iris['data'], sickit_iris['target']],
    columns=np.append(sickit_iris.feature_names, ['y'])
)
print(pd_iris.head(3))

# 选择全部特征参与训练模型
X = pd_iris[sickit_iris.feature_names]
y = pd_iris['y']

# 1. 选择模型
knn = KNeighborsClassifier(n_neighbors=2)

# 2. 拟合模型（训练模型）
knn.fit(X, y)

# 3. 预测新数据，参数为2D数组
result = knn.predict([[5, 3, 4, 3]])
print('result:', result)
