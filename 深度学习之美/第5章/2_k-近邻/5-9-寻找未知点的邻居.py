import operator
import math


# 求欧氏距离函数
def euclidDist(list1: list, list2: list):
    distance = 0.0
    for x in range(len(list1)):
        distance += pow((list1[x] - list2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainSet: [[]], testSet, k):
    # distances = []
    # for x in range(len(trainSet)):
    #     distances.append((trainSet[x], euclidDist(testSet, trainSet[x])))
    # 以上3行等价于以下一行的 列表推导式
    distances = [(trainSet[x], euclidDist(testSet, trainSet[x])) for x in range(len(trainSet))]
    print("before sort: ", distances)
    distances.sort(key=operator.itemgetter(1))
    print("after sort: ", distances)
    return [distances[x][0] for x in range(k)]

trainSet = [[3, 2, 6,'a'], [1, 2, 4, 'b'],[2, 2, 2,'b'],[1, 5, 4,'a']]
testSet = [4, 6, 7]
k = 2
neighbors = getNeighbors(trainSet, testSet, k)
print('测试样本最近的邻居为:', neighbors)
