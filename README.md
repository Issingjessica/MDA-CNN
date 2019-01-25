# MDA-CNN
MDA-CNN

a reference implementation of MDA-CNN as described in the paper<br>
Predicting miRNA-disease association via convolutional neural networks

#####
MDA-CNN is a novel learning-based framework for miRNA-disease association identification. 

it contains three steps.
 
 First, given a three-layer network, we apply a regression model to calculate the disease-gene and miRNA-gene association scores and generate feature vectors for disease and miRNA pairs based on these association scores.
 
 Second, given a pair of miRNA and disease, corresponding feature vector is passed through an auto-encoder-based model to obtain a low dimensional representation.
 
 Third, a deep convolutional neural network (CNN) architecture is constructed to predict the association between miRNA and disease based on the representation vector obtained in last step.

 In the source code folder, the three steps correspond to "regression", "autoencoder" and "CNN" respectively. The user can apply the whole framework to their interested dataset or only use part of the framework. 

 MDA-CNN was implemented with Python 3.6.1. If you have any question or need any technical support, please contact Weiwei Hui (wwhui AT mail DOT nwpu DOT edu DOT cn) and Jiajie Peng (jiajiepeng AT nwpu DOT edu DOT cn). 
 
 
 ####
To run the MDA-CNN, you need these packages:
 networkx
 
 tensorflow
 
 numpy 
 
 sklearn
 
 pandas 
 
 matplotlib


