

import operator


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


neighbors = [['a'], ['b'], ['b']]
type = getClass(neighbors)
print("class: ", type)


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return correct / float(len(testSet))

accuracy = getAccuracy([['a'], ['b'], ['b']], ['a', 'a', 'b'])
print('accuracy: ', accuracy)
