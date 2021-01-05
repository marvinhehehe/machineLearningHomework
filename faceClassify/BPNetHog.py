from numpy import *
import csv
import os
import cv2
from skimage.filters import threshold_otsu
from skimage import feature
from random import randint

testSequence = []
trainLabel = []
trainData = []
testData = []

testSum = 7178
sampleSum = 28709
# sampleSum = 40
input = 0
output = 7
hid = 20
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

    input = 150
    w1 = mat(random.rand(input, hid))
    w2 = mat(random.rand(hid, output))
    for num in range(600):
        print(num)
        for i in range(28709):
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

    ans = {}
    out = open('output.csv', 'a', newline='')

    csv_write = csv.writer(out, dialect='excel')

    for i in range(testSum):
        temp1 = dot(testData[i], w1).T + bias1
        net = sigmoid(temp1)
        temp2 = dot(net.T, w2) + bias2.T
        temp2 = temp2.T
        z = sigmoid(temp2)
        pos = argmax(z)
        if pos == 0:
            ans[testSequence[i]] = "angry"
        if pos == 1:
            ans[testSequence[i]] = "disgust"
        if pos == 2:
            ans[testSequence[i]] = "fear"
        if pos == 3:
            ans[testSequence[i]] = "happy"
        if pos == 4:
            ans[testSequence[i]] = "neutral"
        if pos == 5:
            ans[testSequence[i]] = "sad"
        if pos == 6:
            ans[testSequence[i]] = "surprise"
    test = csv.reader(open("submission.csv", encoding='UTF-8'))
    number = 0
    for row in test:
        if number != 0:
            csv_write.writerow([row[0], ans[row[0]]])
        number = number + 1


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
        efficiencyPos = i + randint(0, 25000)
        temp1 = dot(trainData[efficiencyPos], w1).T + bias1
        net = sigmoid(temp1)
        temp2 = dot(net.T, w2) + bias2.T
        temp2 = temp2.T
        z = sigmoid(temp2)
        pos = argmax(z)
        if pos == 0 and trainLabel[efficiencyPos][0, 0] == 1:
            correct = correct + 1
        if pos == 1 and trainLabel[efficiencyPos][0, 1] == 1:
            correct = correct + 1
        if pos == 2 and trainLabel[efficiencyPos][0, 2] == 1:
            correct = correct + 1
        if pos == 3 and trainLabel[efficiencyPos][0, 3] == 1:
            correct = correct + 1
        if pos == 4 and trainLabel[efficiencyPos][0, 4] == 1:
            correct = correct + 1
        if pos == 5 and trainLabel[efficiencyPos][0, 5] == 1:
            correct = correct + 1
        if pos == 6 and trainLabel[efficiencyPos][0, 6] == 1:
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
    trainData = lowDDataMat[0:28709]
    testData = lowDDataMat[28709:35887]


def robert_filter(image):
    h = image.shape[0]
    w = image.shape[1]
    image_new = zeros(image.shape, uint8)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            image_new[i][j] = abs((image[i][j] - image[i + 1][j + 1])) + abs(image[i + 1][j] - image[i][j + 1])
    return image_new


if __name__ == '__main__':
    for filename in os.listdir(r"./train/angry"):
        position = "train/angry/" + filename
        img1 = cv2.imread(position, cv2.IMREAD_GRAYSCALE)
        features = feature.hog(img1,
                               orientations=9,
                               pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               block_norm='L1',
                               transform_sqrt=True,
                               feature_vector=True)
        trainData.append(mat(array(features)))
        label = [1, 0, 0, 0, 0, 0, 0]
        trainLabel.append(mat(array(label)))
    for filename in os.listdir(r"./train/disgust"):
        position = "train/disgust/" + filename
        img1 = cv2.imread(position, cv2.IMREAD_GRAYSCALE)
        features = feature.hog(img1,
                               orientations=9,
                               pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               block_norm='L1',
                               transform_sqrt=True,
                               feature_vector=True)
        trainData.append(mat(array(features)))
        label = [0, 1, 0, 0, 0, 0, 0]
        trainLabel.append(mat(array(label)))
    for filename in os.listdir(r"./train/fear"):
        position = "train/fear/" + filename
        img1 = cv2.imread(position, cv2.IMREAD_GRAYSCALE)
        features = feature.hog(img1,
                               orientations=9,
                               pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               block_norm='L1',
                               transform_sqrt=True,
                               feature_vector=True)
        trainData.append(mat(array(features)))
        label = [0, 0, 1, 0, 0, 0, 0]
        trainLabel.append(mat(array(label)))
    for filename in os.listdir(r"./train/happy"):
        position = "train/happy/" + filename
        img1 = cv2.imread(position, cv2.IMREAD_GRAYSCALE)
        features = feature.hog(img1,
                               orientations=9,
                               pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               block_norm='L1',
                               transform_sqrt=True,
                               feature_vector=True)
        trainData.append(mat(array(features)))
        label = [0, 0, 0, 1, 0, 0, 0]
        trainLabel.append(mat(array(label)))
    for filename in os.listdir(r"./train/neutral"):
        position = "train/neutral/" + filename
        img1 = cv2.imread(position, cv2.IMREAD_GRAYSCALE)
        features = feature.hog(img1,
                               orientations=9,
                               pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               block_norm='L1',
                               transform_sqrt=True,
                               feature_vector=True)
        trainData.append(mat(array(features)))
        label = [0, 0, 0, 0, 1, 0, 0]
        trainLabel.append(mat(array(label)))
    for filename in os.listdir(r"./train/sad"):
        position = "train/sad/" + filename
        img1 = cv2.imread(position, cv2.IMREAD_GRAYSCALE)
        features = feature.hog(img1,
                               orientations=9,
                               pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               block_norm='L1',
                               transform_sqrt=True,
                               feature_vector=True)
        trainData.append(mat(array(features)))
        label = [0, 0, 0, 0, 0, 1, 0]
        trainLabel.append(mat(array(label)))
    for filename in os.listdir(r"./train/surprise"):
        position = "train/surprise/" + filename
        img1 = cv2.imread(position, cv2.IMREAD_GRAYSCALE)
        features = feature.hog(img1,
                               orientations=9,
                               pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               block_norm='L1',
                               transform_sqrt=True,
                               feature_vector=True)
        trainData.append(mat(array(features)))
        label = [0, 0, 0, 0, 0, 0, 1]
        trainLabel.append(mat(array(label)))
    for filename in os.listdir(r"./test"):
        position = "test/" + filename
        img1 = cv2.imread(position, cv2.IMREAD_GRAYSCALE)
        features = feature.hog(img1,
                               orientations=9,
                               pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               block_norm='L1',
                               transform_sqrt=True,
                               feature_vector=True)
        testData.append(mat(array(features)))
        testSequence.append(filename)
    pca(150)
    BPNet()
    BPtest()
