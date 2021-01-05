import csv
from numpy import *
import cv2
from tensorflow.keras.models import load_model
import os


def output():
    test = csv.reader(open("submission.csv", encoding='UTF-8'))
    out = open('./output.csv', 'w', newline='')
    csv_write = csv.writer(out, dialect='excel')
    number = 0
    for row in test:
        if number != 0:
            csv_write.writerow([row[0], ans[row[0]]])
        number = number + 1



testData = ndarray(shape=[7178, 48, 48])
testSequence = []
number = 0
for filename in os.listdir(r"./test"):
    position = "test/" + filename
    img1 = (cv2.imread(position, cv2.IMREAD_GRAYSCALE))
    testData[number] = img1 / 255.0
    testSequence.append(filename)
    number = number + 1
testData = testData.reshape(testData.shape[0], 48, 48, 1)
model2 = load_model('models/fer_model_finetuned.h5')
haha = model2.predict(testData)
ans = {}

for i in range(7178):
    pos = argmax(haha[i])
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
output()