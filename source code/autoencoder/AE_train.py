import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
import au_calss as au
import pandas as pd
import random as rd
import argparse
import data_helpers as hp



def parse_args():
    parser = argparse.ArgumentParser(description="Run autoencoder.")
    ## the input file
    ##disease-gene relationships and miRNA-gene relatiohships
    parser.add_argument('--input_disease', nargs='?', default='..\..\data\AE\disease_gene.csv',
                        help='Input disease_gene_relationship file')
    parser.add_argument('--input_miRNA', nargs='?', default='..\..\data\AE\miRNA_gene.csv',
                        help = 'Input miRNA_gene_relationship file')
    parser.add_argument('--input_positive',nargs = '?',default='..\..\data\AE\pos.txt',
                        help='positive samples')
    parser.add_argument('--input_negative', nargs='?', default='..\..\data\AE\\neg.txt',
                        help='negative samples')
    parser.add_argument('--output', nargs='?', default='..\..\data\AE\\result\disease_miRNA.csv',
                        help = 'Output low-dimensional disease_miRNA file')
    parser.add_argument('--label_file', nargs='?', default='..\..\data\AE\\result\label.csv',
                        help='Output label file')
    parser.add_argument('--dimensions', nargs='?', default=1024,
                        help ='low dimensional representation')
    parser.add_argument('--batch_size', nargs='?', default=128,
                        help = 'number of samples in one batch')
    parser.add_argument('--training_epochs', nargs='?', default=1,
                        help= 'number of epochs in SGD')
    parser.add_argument('--display_step', nargs='?', default=1)
    parser.add_argument('--input_n_size', nargs='?', default=[3578, 1024])
    parser.add_argument('--hidden_size', nargs='?', default=[1024])
    parser.add_argument('--gene_num', nargs= '?', default = 3578,
                        help= 'number of genes related to disease and miRNA')
    parser.add_argument('--transfer_function', nargs = '?', default= tf.nn.sigmoid,
                        help= 'the activation function')
    parser.add_argument('--optimizer', nargs='?',default= tf.train.AdamOptimizer,
                        help='optimizer for learning weights')
    parser.add_argument('--learning_rate',nargs= '?',default=0.001,
                        help='learning rate for the SGD')
    return parser.parse_args()

def standard_scale(X_train):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    return X_train

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]

def main(args):
    gene_num = args.gene_num
    training_epochs = args.training_epochs
    batch_size = args.batch_size
    display_step = args.display_step
    input_n_size = args.input_n_size
    hidden_size = args.hidden_size
    transfer_function = args.transfer_function
    optimizer = args.optimizer
    learning_rate = args.learning_rate
    label_file = args.label_file
    data, label = hp.get_samples(gene_num, args)
    label = pd.DataFrame(label)
    label.to_csv(label_file, header=None, index= None)
    sdne = []
    ###initialize
    for i in range(len(hidden_size)):
        ae = au.Autoencoder(n_input = input_n_size[i],n_hidden = hidden_size[i],
                            transfer_function = transfer_function,
                            optimizer = optimizer(learning_rate= learning_rate),
                            scale=0)
        sdne.append(ae)
    Hidden_feature = []
    for j in range(len(hidden_size)):
        if j == 0:
            X_train = standard_scale(data)
        else:
            X_train_pre = X_train
            X_train = sdne[j-1].transform(X_train_pre)
            Hidden_feature.append(X_train)

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(X_train.shape[0] / batch_size)

            for batch in range(total_batch):

                batch_xs = get_random_block_from_data(X_train, batch_size)

                cost = sdne[j].partial_fit(batch_xs)
                print("after = %f " % cost)

                avg_cost += cost / X_train.shape[0] * batch_size
            if epoch % display_step == 0:
                print("Epoch:", "%4d" % (epoch + 1), "cost:", "{:.9f}".format(avg_cost))

        if j == 0:
            feat0 = sdne[0].transform(standard_scale(data))
            data1 = pd.DataFrame(feat0)
            data1.T.to_csv(args.output)

if __name__ == "__main__":
    args = parse_args()
    main(args)

