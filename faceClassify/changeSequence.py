import csv
from numpy import *

if __name__ == '__main__':
    angryAns = {}
    disgustAns = {}
    fearAns = {}
    happyAns = {}
    neutralAns = {}
    sadAns = {}
    surpriseAns = {}
    out = open('bag.csv', 'a', newline='')

    csv_write = csv.writer(out, dialect='excel')
    output = csv.reader(open("output.csv", encoding='UTF-8'))
    for row in output:
        if (row[0] not in angryAns):
            angryAns[row[0]] = 0
            disgustAns[row[0]] = 0
            fearAns[row[0]] = 0
            happyAns[row[0]] = 0
            neutralAns[row[0]] = 0
            sadAns[row[0]] = 0
            surpriseAns[row[0]] = 0
        if (row[1] == 'angry'):
            angryAns[row[0]] = angryAns[row[0]] + 1
        if (row[1] == 'disgust'):
            disgustAns[row[0]] = disgustAns[row[0]] + 1
        if (row[1] == 'fear'):
            fearAns[row[0]] = fearAns[row[0]] + 1
        if (row[1] == 'happy'):
            happyAns[row[0]] = happyAns[row[0]] + 1
        if (row[1] == 'neutral'):
            neutralAns[row[0]] = neutralAns[row[0]] + 1
        if (row[1] == 'sad'):
            sadAns[row[0]] = sadAns[row[0]] + 1
        if (row[1] == 'surprise'):
            surpriseAns[row[0]] = surpriseAns[row[0]] + 1
    submission = csv.reader(open("submission.csv", encoding='UTF-8'))
    number = 0
    for row in submission:
        if number != 0:
            pos = argmax([angryAns[row[0]], disgustAns[row[0]], fearAns[row[0]], happyAns[row[0]], neutralAns[row[0]],
                         sadAns[row[0]], surpriseAns[row[0]]])
            if pos == 0:
                csv_write.writerow([row[0], "angry"])
            if pos == 1:
                csv_write.writerow([row[0], "disgust"])
            if pos == 2:
                csv_write.writerow([row[0], "fear"])
            if pos == 3:
                csv_write.writerow([row[0], "happy"])
            if pos == 4:
                csv_write.writerow([row[0], "neutral"])
            if pos == 5:
                csv_write.writerow([row[0], "sad"])
            if pos == 6:
                csv_write.writerow([row[0], "surprise"])
        number = number + 1
