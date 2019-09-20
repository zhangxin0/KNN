import read_data
import numpy as np
import matplotlib.pyplot as plt

data, train_data, validation_data, test_data, y, y_train, y_validation, y_test = read_data.read_train_data()
# data size discription:
print('data size:', len(data))
print('train_data size:', len(train_data))
print('validation_data size:', len(validation_data))
print('test_data size:', len(test_data))
test = [0, 1, 20000, 10000]
i = 0

plt.figure("Digit_show")
for x in test:
    i += 1
    np_data = np.array(data[x], dtype='int')
    np_data = np_data.reshape([28, 28])
    plt.subplot(1, len(test), i)
    plt.imshow(np_data)
plt.show()
