import operator
import csv
import random
import math


# 加载csv数据集
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

# 求欧氏距离函数
def euclidDist(list1: list, list2: list):
    distance = 0.0
    for x in range(len(list1) - 1): # 长度-1是由于最后一列是标签，不参与距离计算
        distance += pow((list1[x] - list2[x]), 2)
    return math.sqrt(distance)

# 获取邻居
def getNeighbors(trainSet: [[]], testSet, k):
    # distances = []
    # for x in range(len(trainSet)):
    #     distances.append((trainSet[x], euclidDist(testSet, trainSet[x])))
    # 以上3行等价于以下一行的 列表推导式
    distances = [(trainSet[x], euclidDist(testSet, trainSet[x])) for x in range(len(trainSet))]
    # print("before sort: ", distances)
    distances.sort(key=operator.itemgetter(1))
    # print("after sort: ", distances)
    return [distances[x][0] for x in range(k)]

# 获取分类
def getClass(neighbors):
    classVotes = {}
    for neighbor in neighbors:
        currentClass = neighbor[-1]
        if currentClass in classVotes:
            classVotes[currentClass] += 1
        else:
            classVotes[currentClass] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

# 获取分类的准确率：分类正确的数量 / 总的测试集样本数
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return correct / float(len(testSet))

def predict(trainSet, testSet, k):
    predictions = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainSet, testSet[x], k)
        classResult = getClass(neighbors)
        predictions.append(classResult)
        print('>预测:', repr(classResult), ', 实际:', testSet[x][-1])
    return predictions

# 主函数
def main():
    filename = 'iris.data'
    trainSet, testSet = loadDataset(filename, 0.7)
    print('训练集样本：' + repr(len(trainSet)) + '，预览：', trainSet)
    print('测试集样本：' + repr(len(testSet)) + '，预览：', testSet)

    k = 3
    predictions = predict(trainSet, testSet, k)
    accuracy = getAccuracy(testSet, predictions)
    print('精确度为: %.3f%%'% (accuracy*100))


main()
