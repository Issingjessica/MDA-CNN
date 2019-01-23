#coding=gbk
import pickle
import math
import numpy as np
from math import isnan

file_path = '..\..\data\disease-miRNA\\'

#将disList读取进来
fDis = open(file_path + 'disList.txt', 'rb')

disList = list()
disList = pickle.load(fDis)



#关闭文件
fDis.close()


#将disList中的disease和其id号关联起来
disId = dict()
i = 0
for dis in disList:
    disId[dis] = i
    i = i + 1


#将geneList读取进来
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


#将计算得到的disease和gene之间的关系读取进来
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


#关闭文件
fDisGene.close()



#计算每个disease term的总和
disSumDict = dict()
for dis in disList:
    disSum = 0
    for gene in geneList:
        disSum += math.exp(disGeneMat[disId[dis]][geneId[gene]])
    disSumDict[dis] = disSum
    



#计算softmax之后的结果，并将结果保存到文件中
fDisGeneSoftmax = open(file_path +'disGeneSoftmax.txt', 'w')

for dis in disList:
    for gene in geneList:
        correSoftmax = math.exp(disGeneMat[disId[dis]][geneId[gene]]) / disSumDict[dis]
        
        fDisGeneSoftmax.write(dis + '\t' + gene + '\t' + str(correSoftmax) + '\n')

#关闭文件
fDisGeneSoftmax.close()

