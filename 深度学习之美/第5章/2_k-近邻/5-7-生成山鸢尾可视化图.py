# import pandas as pd
# https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
import csv
import matplotlib.pyplot as plt


def loadDataset(filename):
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        dataset = list(csv_reader)  # 二维数组，第一维是行，第二维是列
        dataset = [row for row in dataset if row] # 处理空行问题，if row，就是当row为非空的时候返回True
        for i in range(len(dataset)):
            for j in range(4):
                dataset[i][j] = float(dataset[i][j].strip())
    return dataset

filename = 'iris.data'
dataset = loadDataset(filename)
# 花萼长度(sepal_length)、花萼宽度(sepal_width)、花瓣的长度(petal_width)和花瓣的宽度(petal_width)、花的种类标签

# 根据数据的分布，[0,50)为Setosa，[50,100)为Versicolor，[100,150)为Virginica
x1 = [row[0] for row in dataset[0:50]]      # 花萼长度
y1 = [row[2] for row in dataset[0:50]]      # 花瓣的长度
x2 = [row[0] for row in dataset[50:100]]
y2 = [row[2] for row in dataset[50:100]]
x3 = [row[0] for row in dataset[100:]]
y3 = [row[2] for row in dataset[100:]]

# 鸢(yuān)
# 山鸢尾(Iris Setosa)、变色鸢尾(Iris Versicolor)和 维吉尼亚鸢尾(Iris Virginica)
plt.scatter(x1, y1, color='blue', marker='x', label='setosa')       # 山鸢尾
plt.scatter(x2, y2, color='red', marker='o', label='versicolor')    # 变色鸢尾
plt.scatter(x3, y3, color='green', marker='*', label='virginica')   # 维吉尼亚鸢尾
plt.xlabel('petal with')
plt.ylabel('sepal length')
# plt.legend(loc='upper left')
plt.legend()
plt.show()
