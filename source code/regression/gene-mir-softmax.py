#coding=gbk
import pickle
import math
import numpy as np
from math import isnan

file_path = '..\..\data\disease-miRNA\\'
#将mirList中的miRNA加载进来
fMir = open(file_path + 'mirList.txt', 'rb')

mirList = list()
mirList = pickle.load(fMir)



#关闭文件
fMir.close()


#将mirList中的miRNA和其id号关联起来
mirId = dict()
i = 0
for mir in mirList:
    mirId[mir] = i
    i = i + 1


#将geneList中的gene加载进来
fGene = open(file_path +'geneList.txt', 'rb')

geneList = list()
geneList = pickle.load(fGene)



#关闭文件
fGene.close()


#将geneList中的gene和其id号关联起来
geneId = dict()
i = 0
for gene in geneList:
    geneId[gene] = i
    i = i + 1
    

#将miRNA和gene之间的关联关系加载进来
fGeneMir = open(file_path +'geneMirCorre.txt', 'r')

mirGeneMat = np.zeros((243,1789))

while True:
    line = fGeneMir.readline().strip()
    
    if line == '':
        break
    
    terms = line.split('\t')
    gene = terms[0]
    mir = terms[1]
    corre = float(terms[2])
    
    if isnan(corre):
        corre = 0
        
    mirGeneMat[mirId[mir]][geneId[gene]] = corre

#关闭文件
fGeneMir.close()



#计算每个miRNA term的总和
mirSumDict = dict()
for mir in mirList:
    mirSum = 0
    for gene in geneList:
        mirSum += math.exp(mirGeneMat[mirId[mir]][geneId[gene]])
        
    mirSumDict[mir] =  mirSum





#计算softmax之后的结果，并将结果保存到文件中
fMirGeneSoftmax = open(file_path +'mirGeneSoftmax.txt', 'w')

for mir in mirList:
    for gene in geneList:
        correSoftmax = math.exp(mirGeneMat[mirId[mir]][geneId[gene]]) / mirSumDict[mir]
        
        fMirGeneSoftmax.write(mir + '\t' + gene + '\t' + str(correSoftmax) + '\n')

#关闭文件
fMirGeneSoftmax.close()