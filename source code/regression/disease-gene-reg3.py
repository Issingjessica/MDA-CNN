#coding=gbk
import xlrd
import pickle
import networkx as nx
import numpy as np
from sklearn import linear_model

file_path = '..\..\data\disease-miRNA\\'

#��Excel�е�disease���ؽ���
data = xlrd.open_workbook(file_path + 'disease.xlsx')
sheet = data.sheets()[0]
tempList = sheet.col_values(1)

disNameList = list()
for dis in tempList:
    temp = str(dis).lower()
    disNameList.append(temp)





#��diNameList�е�disease����id�Ź�������
disNameId = dict()
i = 0
for dis in disNameList:
    disNameId[dis] = i
    i = i + 1
    
    
#��disease similarity���ؽ���
data = xlrd.open_workbook(file_path +'Gaussian_disease.xlsx')

disNameLen = len(disNameList)
disSim = np.zeros((disNameLen,disNameLen))

sheet = data.sheets()[0]

for i in range(383):
    rowData = sheet.row_values(i)
    for j in range(383):
        disSim[i][j] = rowData[j]
        



#��diseaseList���ؽ���
fDis = open(file_path +'disList.txt', 'rb')

disList = list()
disList = pickle.load(fDis)



#�ر��ļ�
fDis.close()


#��disease�е�disease����id�Ź�������
disId = dict()
i = 0
for dis in disList:
    disId[dis] = i
    i = i + 1



#��gene network���ؽ���
fGeneNet = open(file_path +'GeneNetwork.txt', 'rb')

geneNet = nx.Graph()
geneNet = pickle.load(fGeneNet)

geneNetList = list()
for gene in geneNet:
    if gene not in geneNetList:
        geneNetList.append(gene)
        


#�ر��ļ�
fGeneNet.close()


#��geneList���ؽ���
fGene = open(file_path +'geneList.txt', 'rb')

geneList = list()
geneList = pickle.load(fGene)



#�ر��ļ�
fGene.close()


#��disease��gene�Ĺ�ϵ���ؽ���
fGeneDis = open(file_path +'curated_gene_disease_associations.tsv', 'r')

disGeneDict = dict()

i = 0
while True:
    line = fGeneDis.readline().strip()
    
    if line == '':
        break
    
    if i > 0:
        terms = line.split('\t')
        gene = terms[1]
        dis = terms[3].lower()
        
        if dis not in disList:
            i = i + 1
            continue
        
        if gene not in geneNetList:
            i = i + 1
            continue
        
        if dis not in disGeneDict:
            disGeneDict[dis] = list()
            disGeneDict[dis].append(gene)
        else:
            if gene not in disGeneDict[dis]:
                disGeneDict[dis].append(gene)
                
    i = i + 1
    


#�ر��ļ�
fGeneDis.close()


#���ϲ����geneMergeList�е�gene���ؽ���
fGeneMerge = open(file_path +'geneMergeList.txt', 'rb')

geneMergeList = list()
geneMergeList = pickle.load(fGeneMerge)



#�ر��ļ�
fGeneMerge.close()


#��geneMergeList�е�gene����id�Ź�������
geneMergeId = dict()
i = 0
for gene in geneMergeList:
    geneMergeId[gene] = i
    i = i + 1



#��gene��disease֮���closeness��ϵ���ؽ���
fGeneDisClo = open(file_path +'geneDisClo.txt', 'r')

geneDisCloMat = np.zeros((4277,204))

while True:
    line = fGeneDisClo.readline().strip()
    
    if line == '':
        break
    
    terms = line.split('\t')
    gene = terms[0]
    dis = terms[1]
    clo = float(terms[2])
    
    geneDisCloMat[geneMergeId[gene]][disId[dis]] = clo

#�ر��ļ�
fGeneDisClo.close()



#�����¼����disease similarity��ȡ����
fDisSim = open(file_path +'disRegSim.txt', 'r')

disSimMat = np.zeros((204,204))

while True:
    line = fDisSim.readline().strip()
    
    if line == '':
        break
    
    terms = line.split('\t')
    dis1 = terms[0]
    dis2 = terms[1]
    sim = float(terms[2])
    
    disSimMat[disId[dis1]][disId[dis2]] = sim

#�ر��ļ�
fDisSim.close()


#����disease��gene֮�����ع�ϵ
fDisGeneCorre = open(file_path +'\disGeneCorre.txt', 'w')

i = 0
for dis in disList:
    for gene in geneList:
        
        disSimList = list()
        for dis1 in disList:
            disSimList.append(disSimMat[disId[dis]][disId[dis1]])
            
        geneDisCloList = list()
        for dis2 in disList:
            geneDisCloList.append(geneDisCloMat[geneMergeId[gene]][disId[dis2]])
            
        cov_matrix = np.cov(disSimList, geneDisCloList)
        cov_value = cov_matrix[0][1]
        cov_std = np.sqrt(cov_matrix[0][0] * cov_matrix[1][1])
        cs = cov_value / cov_std
        
        fDisGeneCorre.write(dis + '\t' + gene + '\t' + str(cs) + '\n')
        
        i = i + 1

#�ر��ļ�
fDisGeneCorre.close()

