import numpy as np
import itertools
from collections import Counter
import pandas as pd
import tensorflow as tf





def get_samples(args):
    train_data = pd.read_csv(args.input_disease_miRNA).T
    train_data = np.array(train_data).tolist()
    print('done reading data')
    train_label = pd.read_csv(args.input_label, header=None)
    print("done reading label")
    train_label = np.array(train_label).tolist()
    return train_data, train_label

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
#
# a, b = get_samples()
# # print(len(a[1]))
# # print(len(b))
# # train_label = pd.read_csv('label.csv', header= None)
# # print(np.array(train_label).tolist())
# # print("done reading label")
# # a = np.array(train_label)
# # print(len(a))
# # b = a.tolist()
# #
# # for i in b:
# #     print(i)
# '''input_data, input_label = get_samples()
# print("done loading data")
# dev_sample_percentage = 0.1
# x = np.array(input_data)
# # Randomly shuffle data
# np.random.seed(10)
# batch_size = 120
# num_epochs = 3
# a = len(input_label)
# b = np.arange(len(input_label))
# shuffle_indices = np.random.permutation(b)
# #shuffle_indices = np.random.permutation(np.arange(len(input_label)))
# #input_data = x[shuffle_indices]
# input_data = [x[i] for i in shuffle_indices]
# input_label = [input_label[i] for i in shuffle_indices]
# #input_label = input_label[shuffle_indices]
#
# dev_sample_index = -1 * int(dev_sample_percentage * float(len(input_label)))
# x_train, x_dev = input_data[:dev_sample_index], input_data[dev_sample_index:]
# y_train, y_dev = input_label[:dev_sample_index], input_label[dev_sample_index:]
# print(len(x_train))
# batches = batch_iter(
#             list(zip(x_train, y_train)), batch_size, num_epochs)
# print(len(list(batches)))
# for batch in batches:
#     x,y = zip(*batch)
#     print(len(x))'''

