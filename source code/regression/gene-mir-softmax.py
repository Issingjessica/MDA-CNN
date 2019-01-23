#coding=gbk
import pickle
import math
import numpy as np
from math import isnan

file_path = '..\..\data\disease-miRNA\\'
#��mirList�е�miRNA���ؽ���
fMir = open(file_path + 'mirList.txt', 'rb')

mirList = list()
mirList = pickle.load(fMir)



#�ر��ļ�
fMir.close()


#��mirList�е�miRNA����id�Ź�������
mirId = dict()
i = 0
for mir in mirList:
    mirId[mir] = i
    i = i + 1


#��geneList�е�gene���ؽ���
fGene = open(file_path +'geneList.txt', 'rb')

geneList = list()
geneList = pickle.load(fGene)



#�ر��ļ�
fGene.close()


#��geneList�е�gene����id�Ź�������
geneId = dict()
i = 0
for gene in geneList:
    geneId[gene] = i
    i = i + 1
    

#��miRNA��gene֮��Ĺ�����ϵ���ؽ���
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

#�ر��ļ�
fGeneMir.close()



#����ÿ��miRNA term���ܺ�
mirSumDict = dict()
for mir in mirList:
    mirSum = 0
    for gene in geneList:
        mirSum += math.exp(mirGeneMat[mirId[mir]][geneId[gene]])
        
    mirSumDict[mir] =  mirSum





#����softmax֮��Ľ��������������浽�ļ���
fMirGeneSoftmax = open(file_path +'mirGeneSoftmax.txt', 'w')

for mir in mirList:
    for gene in geneList:
        correSoftmax = math.exp(mirGeneMat[mirId[mir]][geneId[gene]]) / mirSumDict[mir]
        
        fMirGeneSoftmax.write(mir + '\t' + gene + '\t' + str(correSoftmax) + '\n')

#�ر��ļ�
fMirGeneSoftmax.close()