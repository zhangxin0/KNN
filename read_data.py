import csv
import random
"""
read data from csv and transter to list(int)
"""


def transfer_bn(np_data):
    for i in range(len(np_data)):
            if np_data[i] > 0:
                np_data[i] = 1
    return np_data


def read_train_data():
    data = []
    test_data = []
    y = []
    y_test = []
    with open('digit-recognizer/train.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=" ", quotechar='|')
        i = -1
        for row in reader:
            if i == -1:
                i += 1
                continue
            else:
                data.append(transfer_bn(list(map(int, row[0].split(',')[1:]))))
                y.append(int(row[0][0]))
                i += 1
    with open('digit-recognizer/test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=" ", quotechar='|')
        j = -1
        for row in reader:
            if j == -1:
                j += 1
                continue
            else:
                test_data.append(transfer_bn(list(map(int, row[0].split(',')))))
                y_test.append(int(row[0][0]))
                j += 1
    # shuffle data and y with same seed:
    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(data)
    random.seed(randnum)
    random.shuffle(y)
    train_data = data[:int(0.95 * i)]
    y_train = y[:int(0.95 * i)]
    validation_data = data[int(0.95 * i) + 1:]
    y_validation = y[int(0.95 * i) + 1:]
    return data, train_data, validation_data, test_data, y, y_train, y_validation, y_test
