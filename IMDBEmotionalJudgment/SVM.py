import sklearn
from numpy import *
from numpy import *
import csv
from sklearn import svm
import joblib

testSum = 10001
sampleSum = 40001
testSequence = []
trainLabel = []
trainData = []
testData = []
wordStatistic = {}
wordRemain = {}
ans=[]

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

def output():
    out = open('output.csv', 'w', newline='')

    csv_write = csv.writer(out, dialect='excel')
    for i in range(len(ans)):
        if ans[i] == 0:
            csv_write.writerow([testSequence[i], "positive"])
        if ans[i] == 1:
            csv_write.writerow([testSequence[i], "negative"])


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
        if number != 0 and number < testSum:
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
                trainLabel.append(0)
            else:
                trainLabel.append(1)
        number = number + 1
    test = csv.reader(open("test_data.csv", encoding='UTF-8'))
    number = 0
    for row in test:
        if number != 0 and number < testSum:
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
    pca()

    train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(trainData, trainLabel,
                                                                                              random_state=1,
                                                                                              train_size=0.7,
                                                                                              test_size=0.3)
    classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')
    #classifier = svm.SVC(C=2, kernel='poly', decision_function_shape='ovo')
    classifier.fit(train_data, train_label)
    ans = classifier.predict(testData)
    output()
