#coding=gbk
import pickle
import networkx as nx
import numpy as np
import math
from sklearn import linear_model


#将geneList加载进来
fGene = open('geneList.txt', 'rb')

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
fGeneSim = open('geneSim.txt', 'rb')

geneSimMat = pickle.load(fGeneSim)



#关闭文件
fGeneSim.close()



#将miRNA数据加载进来
fMir = open('mirList.txt', 'rb')

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
fMirNet = open('mir_network.txt', 'rb')

mirNet = nx.Graph()
mirNet = pickle.load(fMirNet)

mirNetList = list()
for mir in mirNet:
    if mir not in mirNetList:
        mirNetList.append(mir)
        


#关闭文件
fMirNet.close()


#将gene和miRNA之间的关系加载进来
fGeneMir = open('hsa-vtm-gene.txt', 'r')

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
fMirGeneClo = open('mirGeneClo.txt', 'r')

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



#将根据回归计算的gene similarity重新加载进来
fRegGeneSim = open('regGeneSim.txt', 'r')

regGeneSim = np.zeros((1789,1789))

while True:
    line = fRegGeneSim.readline().strip()
    
    if line == '':
        break
    
    terms = line.split('\t')
    gene1 = terms[0]
    gene2 = terms[1]
    sim = float(terms[2])
    
    regGeneSim[geneId[gene1]][geneId[gene2]] = sim


#关闭文件
fRegGeneSim.close()



#计算miRNA和gene之间的相关关系
fGeneMirCorre = open('geneMirCorre.txt', 'w')


i = 0
for gene in geneList:
    for mir in mirList:
        
        geneSimList = list()
        for gene1 in geneList:
            geneSim = regGeneSim[geneId[gene]][geneId[gene1]]
            geneSimList.append(geneSim)
        
        mirGeneList = list()
        for gene2 in geneList:
            mirGeneClo = mirGeneCloMat[mirId[mir]][geneId[gene2]]
            mirGeneList.append(mirGeneClo)
            
        cov_matrix = np.cov(geneSimList, mirGeneList)
        cov_value = cov_matrix[0][1]
        cov_std = np.sqrt(cov_matrix[0][0] * cov_matrix[1][1])
        cs = cov_value / cov_std
        
        fGeneMirCorre.write(gene + '\t' + mir + '\t' + str(cs) + '\n')
        
        i = i + 1


#关闭文件
fGeneMirCorre.close()
