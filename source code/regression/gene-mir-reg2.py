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







#将计算的回归关系读取进来
fRegCoef = open(file_path +'regCoef.txt', 'r')

regCoefDict = dict()

while True:
    line = fRegCoef.readline().strip()
    
    if line == '':
        break
    
    terms = line.split('\t')
    gene = terms[0]
    mir = terms[1]
    coef = float(terms[2])
    
    regCoefDict[(gene,mir)] = coef

#关闭文件
fRegCoef.close()


#根据回归关系重新计算gene similarity
fRegGeneSim = open(file_path +'regGeneSim.txt', 'w')

i = 0
for gene1 in geneList:
    for gene2 in geneList:
        sim = 0
        for mir in geneMirDict[gene1]:
            sim += regCoefDict[(gene1,mir)] * mirGeneCloMat[mirId[mir]][geneId[gene2]]
        sim += regCoefDict[(gene1,'0')]
        
        fRegGeneSim.write(gene1 + '\t' + gene2 + '\t' + str(sim) + '\n')
        
        i = i + 1


#关闭文件
fRegGeneSim.close()
