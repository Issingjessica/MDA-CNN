#coding=gbk
import pickle
import networkx as nx
import numpy as np
import math
from sklearn import linear_model

file_path = '..\..\data\disease-miRNA\\'

#将geneList加载进来
fGene = open(file_path + 'geneList.txt', 'rb')

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
    

#将gene similarity加载进来
fGeneSim = open(file_path +'geneSim.txt', 'rb')

geneSimMat = pickle.load(fGeneSim)



#关闭文件
fGeneSim.close()



#将miRNA数据加载进来
fMir = open(file_path +'mirList.txt', 'rb')

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
    


#将miRNA network加载进来
fMirNet = open(file_path +'mir_network.txt', 'rb')

mirNet = nx.Graph()
mirNet = pickle.load(fMirNet)

mirNetList = list()
for mir in mirNet:
    if mir not in mirNetList:
        mirNetList.append(mir)
        


#关闭文件
fMirNet.close()


#将gene和miRNA之间的关系加载进来
fGeneMir = open(file_path +'hsa-vtm-gene.txt', 'r')

geneMirDict = dict()

while True:
    line = fGeneMir.readline().strip()
    
    if line == '':
        break
    
    terms = line.split('\t')
    mir = terms[0].lower()
    
    if mir not in mirNetList:
        continue
    
    j = 2
    while j < len(terms):
        gene = terms[j]
        
        if gene not in geneList:
            j = j + 1
            continue
        
        if gene not in geneMirDict:
            geneMirDict[gene] = list()
            geneMirDict[gene].append(mir)
        else:
            if mir not in geneMirDict[gene]:
                geneMirDict[gene].append(mir)
                
        j = j + 1
        



#关闭文件
fGeneMir.close()



#将miRNA和gene之间的closeness关系加载进来
fMirGeneClo = open(file_path +'mirGeneClo.txt', 'r')

mirGeneCloMat = np.zeros((243,1789))

while True:
    line = fMirGeneClo.readline().strip()
    
    if line == '':
        break
    
    terms = line.split('\t')
    mir = terms[0]
    gene = terms[1]
    clo = float(terms[2])
    
    mirGeneCloMat[mirId[mir]][geneId[gene]] = clo


#关闭文件
fMirGeneClo.close()




#构建回归模型，计算gene和miRNA之间的回归关系
fRegCoef = open(file_path +'regCoef.txt', 'w')


count = 0
for gene1 in geneList:
    geneSimList = list()
    regCoefList = list()
    for gene2 in geneList:
        geneSim = geneSimMat[geneId[gene1]][geneId[gene2]]
        geneSimList.append(geneSim)
        
        tempList = list()
        for mir1 in geneMirDict[gene1]:
            mirGeneClo = mirGeneCloMat[mirId[mir1]][geneId[gene2]]
            tempList.append(mirGeneClo)
        regCoefList.append(tempList)
        
    reg = linear_model.LinearRegression()
    reg.fit(regCoefList, geneSimList)
    
    
    i = 0
    for mir2 in geneMirDict[gene1]:
        fRegCoef.write(gene1 + '\t' + mir2 + '\t' + str(reg.coef_[i]) + '\n')
        i = i + 1
        
    fRegCoef.write(gene1 + '\t' + str(0) + '\t' + str(reg.intercept_) + '\n')
    
    count += 1


#关闭文件
fRegCoef.close()
