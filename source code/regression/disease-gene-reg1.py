#coding=gbk
import xlrd
import pickle
import networkx as nx
import numpy as np
from sklearn import linear_model
file_path = '..\..\data\disease-miRNA\\'


#将Excel中的disease加载进来
data = xlrd.open_workbook(file_path +'disease.xlsx')
sheet = data.sheets()[0]
tempList = sheet.col_values(1)

disNameList = list()
for dis in tempList:
    temp = str(dis).lower()
    disNameList.append(temp)





#将diNameList中的disease和其id号关联起来
disNameId = dict()
i = 0
for dis in disNameList:
    disNameId[dis] = i
    i = i + 1
    
    
#将disease similarity加载进来
data = xlrd.open_workbook(file_path + 'Gaussian_disease.xlsx')

disNameLen = len(disNameList)
disSim = np.zeros((disNameLen,disNameLen))

sheet = data.sheets()[0]

for i in range(383):
    rowData = sheet.row_values(i)
    for j in range(383):
        disSim[i][j] = rowData[j]
        



#将diseaseList加载进来
fDis = open(file_path + 'disList.txt', 'rb')

disList = list()
disList = pickle.load(fDis)



#关闭文件
fDis.close()


#将disease中的disease和其id号关联起来
disId = dict()
i = 0
for dis in disList:
    disId[dis] = i
    i = i + 1



#将gene network加载进来
fGeneNet = open(file_path + 'GeneNetwork.txt', 'rb')

geneNet = nx.Graph()
geneNet = pickle.load(fGeneNet)

geneNetList = list()
for gene in geneNet:
    if gene not in geneNetList:
        geneNetList.append(gene)
        


#关闭文件
fGeneNet.close()


#将geneList加载进来
fGene = open(file_path + 'geneList.txt', 'rb')

geneList = list()
geneList = pickle.load(fGene)



#关闭文件
fGene.close()


#将disease和gene的关系加载进来
fGeneDis = open(file_path + 'curated_gene_disease_associations.tsv', 'r')

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
    


#关闭文件
fGeneDis.close()


#将合并后的geneMergeList中的gene加载进来
fGeneMerge = open(file_path + 'geneMergeList.txt', 'rb')

geneMergeList = list()
geneMergeList = pickle.load(fGeneMerge)



#关闭文件
fGeneMerge.close()


#将geneMergeList中的gene和其id号关联起来
geneMergeId = dict()
i = 0
for gene in geneMergeList:
    geneMergeId[gene] = i
    i = i + 1



#将gene和disease之间的closeness关系加载进来
fGeneDisClo = open(file_path + 'geneDisClo.txt', 'r')

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

#关闭文件
fGeneDisClo.close()



#利用回归模型计算回归系数,并将计算的结果保存到文件中
fRegCof = open(file_path + 'regCof.txt', 'w')

count = 0
for dis1 in disList:
    disSimList = list()
    regCoefList = list()
    for dis2 in disList:
        disRow = disNameList.index(dis1)
        disCol = disNameList.index(dis2)
        
        sim = disSim[disRow][disCol]
        disSimList.append(sim)
        
        tempList = list()
        for gene1 in disGeneDict[dis1]:
            tempCoef = geneDisCloMat[geneMergeId[gene1]][disId[dis2]]
            tempList.append(tempCoef)
        regCoefList.append(tempList)
            
    reg = linear_model.LinearRegression()
    reg.fit(regCoefList, disSimList)
    
    i = 0
    for gene2 in disGeneDict[dis1]:
        fRegCof.write(dis1 + '\t' + gene2 + '\t' + str(reg.coef_[i]) + '\n')
        i = i + 1
    
    fRegCof.write(dis1 + '\t' + str(0) + '\t' + str(reg.intercept_) + '\n')
    
    count += 1


#关闭文件
fRegCof.close()

