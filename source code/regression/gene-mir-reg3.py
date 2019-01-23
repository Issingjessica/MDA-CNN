#coding=gbk
import pickle
import networkx as nx
import numpy as np
import math
from sklearn import linear_model


#��geneList���ؽ���
fGene = open('geneList.txt', 'rb')

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
    

#��gene similarity���ؽ���
fGeneSim = open('geneSim.txt', 'rb')

geneSimMat = pickle.load(fGeneSim)



#�ر��ļ�
fGeneSim.close()



#��miRNA���ݼ��ؽ���
fMir = open('mirList.txt', 'rb')

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
    


#��miRNA network���ؽ���
fMirNet = open('mir_network.txt', 'rb')

mirNet = nx.Graph()
mirNet = pickle.load(fMirNet)

mirNetList = list()
for mir in mirNet:
    if mir not in mirNetList:
        mirNetList.append(mir)
        


#�ر��ļ�
fMirNet.close()


#��gene��miRNA֮��Ĺ�ϵ���ؽ���
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
        



#�ر��ļ�
fGeneMir.close()



#��miRNA��gene֮���closeness��ϵ���ؽ���
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


#�ر��ļ�
fMirGeneClo.close()



#�����ݻع�����gene similarity���¼��ؽ���
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


#�ر��ļ�
fRegGeneSim.close()



#����miRNA��gene֮�����ع�ϵ
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


#�ر��ļ�
fGeneMirCorre.close()
