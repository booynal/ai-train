import csv
import random


def loadDataset(filename, trainSplit, trainSet=[], testSet=[]):
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        dataset = list(csv_reader)  # 二维数组，第一维是行，第二维是列
        for i in range(len(dataset) - 1):
            for j in range(4):
                dataset[i][j] = float(dataset[i][j].strip())
            if random.random() < trainSplit:
                trainSet.append(dataset[i])
            else:
                testSet.append(dataset[i])
    return trainSet, testSet

filename = 'iris.data'
trainSet, testSet = loadDataset(filename, 0.7)
print('训练集样本：' + repr(len(trainSet)) + ', 预览：',  trainSet)
print('测试集样本：' + repr(len(testSet)) + '，预览：', testSet)

