import numpy as np
import itertools
from collections import Counter
import sys, re
import pandas as pd
import tensorflow as tf





def get_samples(num_gene, args):
    samples = []
    pos_file = args.input_positive
    neg_file = args.input_negative
    label = []
    disease = []
    micro = []
    with open(pos_file, "r") as f:
        for line in f:
            if line[0] ==' ':
                continue
            line_data = line.strip().split('\t')
            l = line_data[1]
            disease.append(line_data[0])
            micro.append(line_data[1])
            samples.append((line_data[0],line_data[1]))
            label.append([0, 1])

    with open(neg_file, "r") as f:
        for line in f:
            if line[0]==' ':
                continue
            line_data = line.strip().split('\t')

            samples.append((line_data[0],line_data[1]))
            disease.append(line_data[0])
            micro.append(line_data[1])
            label.append([1, 0])



    disease_vector = pd.read_csv(args.input_disease)
    miro_vector = pd.read_csv(args.input_miRNA)
    vocab_size = len(samples)

    W = np.zeros(shape=(vocab_size, num_gene), dtype='float32')
    W[0] = np.zeros(num_gene, dtype='float32')
    i = 0
    for sample in samples:
        v1 = list(disease_vector[sample[0]])
        v2 = list(miro_vector[sample[1]])

        v1.extend(v2)
        W[i] = v1
        i = i + 1
    return W, label



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
'''input_data, input_label = get_samples()
print("done loading data")
dev_sample_percentage = 0.1
x = np.array(input_data)
# Randomly shuffle data
np.random.seed(10)
batch_size = 120
num_epochs = 3
a = len(input_label)
b = np.arange(len(input_label))
shuffle_indices = np.random.permutation(b)
#shuffle_indices = np.random.permutation(np.arange(len(input_label)))
#input_data = x[shuffle_indices]
input_data = [x[i] for i in shuffle_indices]
input_label = [input_label[i] for i in shuffle_indices]
#input_label = input_label[shuffle_indices]

dev_sample_index = -1 * int(dev_sample_percentage * float(len(input_label)))
x_train, x_dev = input_data[:dev_sample_index], input_data[dev_sample_index:]
y_train, y_dev = input_label[:dev_sample_index], input_label[dev_sample_index:]
print(len(x_train))
batches = batch_iter(
            list(zip(x_train, y_train)), batch_size, num_epochs)
print(len(list(batches)))
for batch in batches:
    x,y = zip(*batch)
    print(len(x))'''

