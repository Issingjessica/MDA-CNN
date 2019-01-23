 #! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import argparse
import data_helpers as dh
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from tensorflow.contrib import learn
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
from sklearn.metrics import average_precision_score





def parse_args():
    parser = argparse.ArgumentParser(description="Run CNN.")
    ## the input file
    ##disease-gene relationships and miRNA-gene relatiohships
    parser.add_argument('--input_disease_miRNA', nargs='?', default='..\..\data\CNN\disease-miro-1024-sigmoid.csv',
                        help='Input disease_gene_relationship file')

    parser.add_argument('--input_label',nargs = '?',default='..\..\data\CNN\label.csv',
                        help='sample label')
    parser.add_argument('--batch_size', nargs='?', default=64,
                        help = 'number of samples in one batch')
    parser.add_argument('--training_epochs', nargs='?', default=1,
                        help= 'number of epochs in SGD')
    parser.add_argument('--display_step', nargs='?', default=10)
    parser.add_argument('--test_percentage', nargs='?', default=0.1,
                        help='percentage of test samples')
    parser.add_argument('--dev_percentage', nargs='?', default=0.1,
                        help='percentage of validation samples')
    parser.add_argument('--L2_norm', nargs='?', default=0.001,
                        help='percentage of validation samples')
    parser.add_argument('--keep_prob', nargs='?', default=0.5,
                        help='keep_prob when using dropout option')
    parser.add_argument('--optimizer', nargs='?', default=tf.train.AdamOptimizer,
                        help='optimizer for learning weights')
    parser.add_argument('--learning_rate', nargs='?', default=1e-3,
                        help='learning rate for the SGD')

    return parser.parse_args()
def standard_scale(X_train):

    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)

    return X_train
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev= 0.1)
    weights = tf.Variable(initial)

    return weights
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding= "VALID")
def max_pool_2(x, W):
    return tf.nn.max_pool(x, ksize = W, strides= [1,10,1,1], padding= "VALID")


def get_data(args):

    input_data, input_label = dh.get_samples(args)
    input_data = standard_scale(input_data)
    dev_sample_percentage = args.dev_percentage
    test_sample_percentage = args.test_percentage
    x = np.array(input_data)
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(input_label)))
    input_data = [x[i] for i in shuffle_indices]
    input_label = [input_label[i] for i in shuffle_indices]
    dev_sample_index = -2 * int(dev_sample_percentage * float(len(input_label)))
    test_sample_index = -1 * int(test_sample_percentage * float(len(input_label)))
    x_train, x_dev, test_data = input_data[:dev_sample_index], input_data[dev_sample_index:test_sample_index], input_data[test_sample_index:]
    y_train, y_dev, test_label = input_label[:dev_sample_index], input_label[dev_sample_index:test_sample_index], input_label[test_sample_index:]

    return x_train, x_dev, test_data, y_train, y_dev, test_label


def deepnn(x, keep_prob, args):
    with tf.name_scope('reshape'):
        x = tf.reshape(x, [-1, 1024, 1, 1])

    with tf.name_scope('conv_pool'):
        filter_shape = [4, 1, 1, 4]

        W_conv = weight_variable(filter_shape)
        b_conv = bias_variable([4])
        h_conv = tf.nn.relu(conv2d(x, W_conv) + b_conv)
        h_pool = tf.nn.max_pool(h_conv, ksize = [1, 4, 1, 1], strides= [1,4,1,1], padding= "VALID")

        '''filter_shape2 = [4,1,4,4]
        W_conv2 = weight_variable(filter_shape2)
        b_conv2 = bias_variable([4])
        h_conv2 = tf.nn.relu(conv2d(h_pool, W_conv2) + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,4,1,1], strides= [1,4,1,1],padding="VALID")'''

        regula = tf.contrib.layers.l2_regularizer(args.L2_norm)
        h_input1 = tf.reshape(h_pool,[-1, 255 * 4])
        W_fc1 = weight_variable([255 * 4, 50])

        b_fc1 = bias_variable([50])
        h_input2 = tf.nn.relu(tf.matmul(h_input1, W_fc1) + b_fc1)
        h_keep = tf.nn.dropout(h_input2, keep_prob)
        W_fc2 = weight_variable([50, 2])
        b_fc2 = bias_variable([2])
        h_output = tf.matmul(h_keep, W_fc2) + b_fc2
        regularizer = regula(W_fc1) + regula(W_fc2)
        return h_output, regularizer


def main(args):
    with tf.device('/cpu:0'):
        x_train, x_dev, test_data, y_train, y_dev, test_label = get_data(args)
        input_data = tf.placeholder(tf.float32, [None, 1024])
        input_label = tf.placeholder(tf.float32, [None, 2])
        keep_prob = tf.placeholder(tf.float32)
        y_conv, losses = deepnn(input_data, keep_prob, args)
        y_res = tf.nn.softmax(y_conv)
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=input_label)
        cross_entropy = tf.reduce_mean(cross_entropy)
        los = cross_entropy + losses
        with tf.name_scope('optimizer'):
            optimizer = args.optimizer
            learning_rate = args.learning_rate
            train_step = optimizer(learning_rate).minimize(los)
            #optimizer = tf.train.MomentumOptimizer(learning_rate= 0.02, momentum=)
            #train_step = optimizer.minimize(cross_entropy)
        with tf.name_scope('accuracy'):
            predictions = tf.argmax(y_conv, 1)
            correct_predictions = tf.equal(predictions, tf.argmax(input_label, 1))
            correct_predictions = tf.cast(correct_predictions, tf.float32)
        accuracy = tf.reduce_mean(correct_predictions)

        batch_size = args.batch_size
        num_epochs = args.training_epochs
        display_step = args.display_step
        k_p = args.keep_prob
        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())


                batches = dh.batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)

                for i, batch in enumerate(batches):
                    x_batch, y_batch = zip(*batch)



                    train_step.run(feed_dict={input_data: x_batch, input_label:y_batch, keep_prob :k_p} )


                    if i % display_step == 0:


                        loss = sess.run(los, feed_dict={input_data: x_train, input_label: y_train, keep_prob :1.0})
                        # print('after training loss = %f' % loss)
                        y_predict = sess.run(y_res, feed_dict={input_data: x_dev, input_label: y_dev, keep_prob :1.0})[:, 1]

                        loss = sess.run(los, feed_dict={input_data: x_dev, input_label: y_dev, keep_prob :1.0})
                        # print('test loss = %f' % loss)

                        false_positive_rate1, true_positive_rate1, thresholds1 = roc_curve(np.array(y_dev)[:, 1], y_predict)
                        roc_auc1 = auc(false_positive_rate1, true_positive_rate1)
                        # print(roc_auc1)
                print(accuracy.eval(feed_dict = {input_data: test_data, input_label: test_label, keep_prob :1.0}))

                y_predict = sess.run(y_res, feed_dict={input_data: test_data, input_label: test_label, keep_prob :1.0})[:, 1]
                false_positive_rate1, true_positive_rate1, thresholds1 = roc_curve(np.array(test_label)[:,1], y_predict)
                roc_auc1 = auc(false_positive_rate1, true_positive_rate1)
                print(roc_auc1)
                # np.savetxt("result_fp_tp_md_aver.txt", roc_curve(np.array(test_label)[:, 1], y_predict))
                # precision, recall ,_ = precision_recall_curve(np.array(test_label)[:, 1], y_predict)
                #
                # average_precision = average_precision_score(np.array(test_label)[:, 1], y_predict)
                #
                # print('Average precision-recall score: {0:0.2f}'.format(average_precision))
                # y_predict[y_predict >= 0.5] = 1
                # y_predict[y_predict < 0.5] = 0
                # print(y_predict)
                # print(metrics.f1_score(np.array(test_label)[:, 1], y_predict))
                # np.savetxt("precision_aver.txt", precision)
                # np.savetxt("recall_aver.txt", recall)




    



if __name__ == '__main__':
    args = parse_args()
    main(args)


    












