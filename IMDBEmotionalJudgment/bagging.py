from numpy import *
import csv
from random import randint

testSequence = []
trainLabel = []
trainData = []
testData = []
wordStatistic = {}
wordRemain = {}

randomTrainData = []
randomTrainLabel = []

testSum = 10000
sampleSum = 40001
input = 0
output = 2
hid = 10
w1 = []
w2 = []
bias1 = []
bias2 = []
modelNumber = 0
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
    global modelNumber

    global temp1
    global net
    global temp2
    global z

    input = 650
    w1.append(mat(random.rand(input, hid)))
    w2.append(mat(random.rand(hid, output)))
    bias1.append(mat(zeros((hid, 1))))
    bias2.append(mat(zeros((output, 1))))
    for num in range(200):
        print(num)
        for i in range(2000):
            label = randomTrainLabel[i]  # 暂赋值为0，之后根据训练集的结果赋值
            temp1 = dot(randomTrainData[i], w1[modelNumber]).T + bias1[modelNumber]
            net = sigmoid(temp1)
            temp2 = dot(net.T, w2[modelNumber]) + bias2[modelNumber].T
            temp2 = temp2.T
            z = sigmoid(temp2)
            error = label.T - z
            deltaZ = multiply(multiply(z, (1 - z)), error)
            deltaNet = multiply(multiply(net, (1 - net)), dot(w2[modelNumber], deltaZ))
            w2[modelNumber] = w2[modelNumber] + dot(net, multiply(rate, deltaZ).T)
            w1[modelNumber] = w1[modelNumber] + dot(multiply(rate, deltaNet), randomTrainData[i]).T
            bias1[modelNumber] = bias1[modelNumber] + rate * deltaNet
            bias2[modelNumber] = bias2[modelNumber] + rate * deltaZ
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
    global modelNumber

    global temp1
    global net
    global temp2
    global z

    out = open('output.csv', 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    for i in range(testSum):
        positiveNumber = 0
        negativeNumber = 0
        for time in range(20):
            temp1 = dot(testData[i], w1[time]).T + bias1[time]
            net = sigmoid(temp1)
            temp2 = dot(net.T, w2[time]) + bias2[time].T
            temp2 = temp2.T
            z = sigmoid(temp2)
            if z[0] > z[1]:
                positiveNumber = positiveNumber + 1
            if z[0] <= z[1]:
                negativeNumber = negativeNumber + 1
        if positiveNumber > negativeNumber:
            csv_write.writerow([testSequence[i], "positive"])
        if positiveNumber <= negativeNumber:
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
        positiveNumber = 0
        negativeNumber = 0
        efficiencyPos = i + randint(0, 35000)
        for time in range(modelNumber + 1):
            temp1 = dot(trainData[efficiencyPos], w1[time]).T + bias1[time]
            net = sigmoid(temp1)
            temp2 = dot(net.T, w2[time]) + bias2[time].T
            temp2 = temp2.T
            z = sigmoid(temp2)
            if z[0] > z[1]:
                positiveNumber = positiveNumber + 1
            if z[0] <= z[1]:
                negativeNumber = negativeNumber + 1
        if positiveNumber > negativeNumber and trainLabel[efficiencyPos][0, 0] == 1:
            correct = correct + 1
        if positiveNumber <= negativeNumber and trainLabel[efficiencyPos][0, 1] == 1:
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
    # print(eigVals)
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
    for time in range(20):
        randomTrainData.clear()
        randomTrainLabel.clear()
        for modelTime in range(2000):
            pos = randint(0, 39998)
            randomTrainLabel.append(trainLabel[pos])
            randomTrainData.append(trainData[pos])
        BPNet()
        modelNumber = modelNumber + 1
    BPtest()
