import read_data
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import csv

# np.random.randint(low, high, size) 返回随机的整数，位于半开区间 [low, high)
# np.random.shuffle(x) 类似洗牌，打乱顺序；np.random.permutation(x)返回一个随机排列
"""
KNN Algorithm pseudocode:
For data in test_data:
	Calculate the distance of data to all imgs in train_data
	Get the 7 items with lowest distance
    If only one img has the largest number:
        Choose the label of img with most amount as prediction
    Else multiple imgs have the same largest number:
		Choose the one with smallest distance as prediction

"""
data, train_data, validation_data, test_data, y, y_train, y_validation, y_test = read_data.read_train_data()


def distance(data1, data2):
    sum = 0.
    for i in range(len(data1)):
        sum += math.pow(data1[i] - data2[i], 2)
    return math.sqrt(sum)


def train_model(train_data, y_train, validation_data, y_validation):
    # distance matrix:
    res = []
    # for input data: calculate the distance among all figs
    # for i in validation_data:
    num = 100  # len(y_test)
    monitor = 0
    # 选取 1000 张作为训练数据
    # 选取100张 0，100张 1， 100张 2..
    dict = {}
    counter = 0
    visited = set()
    id = np.random.randint(0, 39900)
    train_data_sample = []  # train data
    y_train_sample = []  # label data
    k = 7
    while counter < 1000:
        id = np.random.randint(0, 39900)
        if id not in visited:
            if y_train[id] not in dict:
                counter += 1
                dict[y_train[id]] = 1
                train_data_sample.append(train_data[id])
                y_train_sample.append(y_train[id])
            elif dict[y_train[id]] < 100:
                counter += 1
                dict[y_train[id]] += 1
                train_data_sample.append(train_data[id])
                y_train_sample.append(y_train[id])
            visited.add(id)
    right_num = 0
    validated = {}
    for i in range(num):
        validate_id = np.random.randint(2099)
        i = validate_id
        # monitor:
        print(monitor)
        monitor += 1
        dis = []  # store (digit label, distance)
        for j in range(len(train_data_sample)):
            dis.append((y_train_sample[j], distance(validation_data[i], train_data_sample[j])))
        dis = sorted(dis, key=lambda item: item[1])
        # get the min 7 distance figs:
        neighbors = dis[:k]
        # dict (y,(nums,Min_dis))
        dict = {}
        vote = []
        for value, dis in neighbors:
            if value not in dict:
                dict[value] = [1, dis]
            else:
                dict[value][0] += 1
                if dis < dict[value][1]:
                    dict[value][1] = dis
        # 选择投票数最多：vote[y,[nums,Min_dis]] dict_items = tuple(key,[value])
        vote = sorted(dict.items(), key=lambda item: item[1][0])
        vt_num_max = vote[-1][1][0]
        # if tie: choose the nearst one 先按照数量排序，再按照距离最近排序
        min_dis = vote[-1][1][1]
        best_predict = vote[-1][0]
        for value in dict:
            if dict[value][0] == vt_num_max and dict[value][1] < min_dis:
                min_dis = dict[value][1]
                best_predict = value
        res.append(best_predict)
        if best_predict == y_validation[i]:
            right_num += 1
    accuracy = right_num / num * 100
    print(res)
    print('Accuracy: ', accuracy, '%')
    # calculate prediction accuracy
    return accuracy, train_data_sample, y_train_sample


epoch = 10
best_accuracy = 0
for i in range(epoch):
    print('epoch:', i)
    accuracy, train_data_model, y_train_model = train_model(train_data, y_train, validation_data, y_validation)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # output train_data, y_train:
        with open('model_data.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(train_data_model)
        with open('model_label.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            # data automatically write back with ',' split
            writer.writerow(y_train_model)
print(best_accuracy)
