from numpy import *
import csv
from random import randint

testSequence = []
trainLabel = []
trainData = []
testData = []
wordStatistic = {}
wordRemain = {}

testSum = 10000
sampleSum = 40001
# sampleSum = 40
input = 0
output = 2
hid = 100
w1 = 0
w2 = 0
bias1 = mat(zeros((hid, 1)))
bias2 = mat(zeros((output, 1)))
rate = 0.1

temp1 = mat(zeros((hid, 1)))
net = temp1
temp2 = mat(zeros((output, 1)))
z = temp2


def sigmoid(x):
    return 1 / (1 + exp(-x))


def BPNet():
    global testSum
    global sampleSum
    global input
    global output
    global hid
    global w1
    global w2
    global bias1
    global bias2
    global rate

    global temp1
    global net
    global temp2
    global z

    input = 650
    w1 = mat(random.rand(input, hid))
    w2 = mat(random.rand(hid, output))
    for num in range(30):
        print(num)
        for i in range(40000):
            label = trainLabel[i]  # 暂赋值为0，之后根据训练集的结果赋值
            temp1 = dot(trainData[i], w1).T + bias1
            net = sigmoid(temp1)
            temp2 = dot(net.T, w2) + bias2.T
            temp2 = temp2.T
            z = sigmoid(temp2)
            error = label.T - z
            deltaZ = multiply(multiply(z, (1 - z)), error)
            deltaNet = multiply(multiply(net, (1 - net)), dot(w2, deltaZ))
            w2 = w2 + dot(net, multiply(rate, deltaZ).T)
            w1 = w1 + dot(multiply(rate, deltaNet), trainData[i]).T
            bias1 = bias1 + rate * deltaNet
            bias2 = bias2 + rate * deltaZ
        learnEfficiencyCheck()


def BPtest():
    global testSum
    global sampleSum
    global input
    global output
    global hid
    global w1
    global w2
    global bias1
    global bias2
    global rate

    global temp1
    global net
    global temp2
    global z

    out = open('output.csv', 'a', newline='')

    csv_write = csv.writer(out, dialect='excel')

    for i in range(testSum):
        temp1 = dot(testData[i], w1).T + bias1
        net = sigmoid(temp1)
        temp2 = dot(net.T, w2) + bias2.T
        temp2 = temp2.T
        z = sigmoid(temp2)
        if z[0] > z[1]:
            csv_write.writerow([testSequence[i], "positive"])
        if z[0] <= z[1]:
            csv_write.writerow([testSequence[i], "negative"])


def learnEfficiencyCheck():
    global testSum
    global sampleSum
    global input
    global output
    global hid
    global w1
    global w2
    global bias1
    global bias2
    global rate

    global temp1
    global net
    global temp2
    global z

    correct = 0
    for i in range(1000):
        efficiencyPos = i + randint(0, 35000)
        temp1 = dot(trainData[efficiencyPos], w1).T + bias1
        net = sigmoid(temp1)
        temp2 = dot(net.T, w2) + bias2.T
        temp2 = temp2.T
        z = sigmoid(temp2)
        if z[0] > z[1] and trainLabel[efficiencyPos][0, 0] == 1:
            correct = correct + 1
        if z[0] <= z[1] and trainLabel[efficiencyPos][0, 1] == 1:
            correct = correct + 1
    print(correct / 1000)


def pca(topNfeat=1000000):
    global trainData
    global testData
    totalMat = mat(array(trainData + testData))
    meanVals = mean(totalMat, axis=0)
    DataAdjust = totalMat - meanVals
    covMat = cov(DataAdjust, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    #print(eigVals)
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = DataAdjust * redEigVects
    # reconMat = (lowDDataMat * redEigVects.T) + meanVals
    trainData = lowDDataMat[0:40000]
    testData = lowDDataMat[40000:50000]


if __name__ == '__main__':
    train = csv.reader(open("train.csv", encoding='UTF-8'))
    number = 0
    for row in train:
        if number != 0 and number < sampleSum:
            splitWord = row[1].split()
            for word in splitWord:
                word = "".join(filter(str.isalpha, word))
                word = word.lower()
                if (word != '<br />'):
                    if (word in wordStatistic.keys()):
                        wordStatistic[word] = 1 + wordStatistic[word]
                    else:
                        wordStatistic[word] = 1
        number = number + 1

    test = csv.reader(open("test_data.csv", encoding='UTF-8'))
    number = 0
    for row in test:
        if number != 0:
            splitWord = row[1].split()
            for word in splitWord:
                word = "".join(filter(str.isalpha, word))
                word = word.lower()
                if (word != '<br />'):
                    if (word in wordStatistic.keys()):
                        wordStatistic[word] = 1 + wordStatistic[word]
                    else:
                        wordStatistic[word] = 1
        number = number + 1

    position = 0
    for word in wordStatistic.keys():
        if wordStatistic[word] > 50:
            wordRemain[word] = position
            position = position + 1
    train = csv.reader(open("train.csv", encoding='UTF-8'))
    number = 0
    for row in train:
        if number != 0 and number < sampleSum:
            splitWord = row[1].split()
            vector = [0] * len(wordRemain)
            for word in splitWord:
                word = "".join(filter(str.isalpha, word))
                word = word.lower()
                if (word in wordRemain.keys()):
                    vector[wordRemain[word]] = 1 + vector[wordRemain[word]]
            maxNumber = max(vector)
            for specialNumber in range(len(vector)):
                if maxNumber != 0:
                    vector[specialNumber] = vector[specialNumber] / maxNumber
            trainData.append(mat(array(vector)))
            if row[2] == 'positive':
                trainLabel.append(mat(array([1, 0])))
            else:
                trainLabel.append(mat(array([0, 1])))
        number = number + 1
    test = csv.reader(open("test_data.csv", encoding='UTF-8'))
    number = 0
    for row in test:
        if number != 0:
            testSequence.append(row[0])
            splitWord = row[1].split()
            vector = [0] * len(wordRemain)
            for word in splitWord:
                word = "".join(filter(str.isalpha, word))
                word = word.lower()
                if (word in wordRemain.keys()):
                    vector[wordRemain[word]] = 1 + vector[wordRemain[word]]
            maxNumber = max(vector)
            for specialNumber in range(len(vector)):
                if maxNumber != 0:
                    vector[specialNumber] = vector[specialNumber] / maxNumber
            testData.append(mat(array(vector)))
        number = number + 1
    pca(650)
    BPNet()
    BPtest()
