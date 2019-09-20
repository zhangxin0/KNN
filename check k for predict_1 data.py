import read_data
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import csv

"""
This file is to plot the relationship between k and prediciton:
"""


def distance(data1, data2):
    sum = 0.
    for i in range(len(data1)):
        sum += math.pow(data1[i] - data2[i], 2)
    return math.sqrt(sum)


# read data:
data, train_data, validation_data, test_data, y, y_train, y_validation, y_test = read_data.read_train_data()

res = []
train_data_sample = []
y_train_sample = []
# read data from csv files:
with open('model_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=" ", quotechar='|')
    for row in reader:
        train_data_sample.append((list(map(int, row[0].split(',')))))
with open('model_label.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=" ", quotechar='|')
    for row in reader:
        # only 1 row:
        y_train_sample = (list(map(int, row[0].split(','))))

# KNN:
rand_num = 0
while y_validation[rand_num] != 5:
    rand_num = np.random.randint(100)
monitor = 0
for k in range(1, 100):
    right_num = 0
    validated = {}
    num = 1
    for i in range(num):
        # monitor:
        i = rand_num
        print(monitor)
        monitor += 1
        dis = []  # store (digit label, distance)
        for j in range(len(train_data_sample)):
            dis.append((y_train_sample[j], distance(validation_data[i], train_data_sample[j])))
        dis = sorted(dis, key=lambda item: item[1])
        # get the min 7 distance figs:
        k = k
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

print(res)
print(y_validation[rand_num])

plt.figure('Test')
x = np.linspace(0, 100, 99)
y = np.array(res)
plt.plot(x, y)
plt.show()
