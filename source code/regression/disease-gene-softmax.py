#coding=gbk
import pickle
import math
import numpy as np
from math import isnan

file_path = '..\..\data\disease-miRNA\\'

#��disList��ȡ����
fDis = open(file_path + 'disList.txt', 'rb')

disList = list()
disList = pickle.load(fDis)



#�ر��ļ�
fDis.close()


#��disList�е�disease����id�Ź�������
disId = dict()
i = 0
for dis in disList:
    disId[dis] = i
    i = i + 1


#��geneList��ȡ����
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


#������õ���disease��gene֮��Ĺ�ϵ��ȡ����
fDisGene = open(file_path +'disGeneCorre.txt', 'r')

disGeneMat = np.zeros((204,1789))

while True:
    line = fDisGene.readline().strip()
    
    if line == '':
        break
    
    terms = line.split('\t')
    dis = terms[0]
    gene = terms[1]
    corre = float(terms[2])
    
    if isnan(corre):
        corre = 0
    
    disGeneMat[disId[dis]][geneId[gene]] = corre


#�ر��ļ�
fDisGene.close()



#����ÿ��disease term���ܺ�
disSumDict = dict()
for dis in disList:
    disSum = 0
    for gene in geneList:
        disSum += math.exp(disGeneMat[disId[dis]][geneId[gene]])
    disSumDict[dis] = disSum
    



#����softmax֮��Ľ��������������浽�ļ���
fDisGeneSoftmax = open(file_path +'disGeneSoftmax.txt', 'w')

for dis in disList:
    for gene in geneList:
        correSoftmax = math.exp(disGeneMat[disId[dis]][geneId[gene]]) / disSumDict[dis]
        
        fDisGeneSoftmax.write(dis + '\t' + gene + '\t' + str(correSoftmax) + '\n')

#�ر��ļ�
fDisGeneSoftmax.close()

