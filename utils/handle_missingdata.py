#@Time:2019/12/14 11:03
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:handle_missingdata.py
__author__ = "skux liu"
import re
import random
import numpy as np
import pandas as pd
from utils.read_file import readNex,saveNex
def hasNan(li):
    for i in li:
        if np.isnan(i):
            return True
    return False
def gene_missingdata(rate,data,seed=0):
    '''
    :param rate: 缺失率，int
    :param data: 完整数据集，np.array
    随机模式缺失
    :return:缺失后数据集和缺失行数
    '''
    #random.seed=seed
    missData=data.copy()
    nrow, ncol = missData.shape
    myList = list(np.linspace(0, ncol*nrow-1, ncol*nrow, dtype=int))
    choice_num=random.sample(myList,int(ncol*nrow*rate))
    for num in choice_num:
        row=num%nrow
        col=num//nrow
        missData[row][col] = np.nan
    return missData
def gene_missingdata_taxa_bias(rate,data,seed=0):
    #random.seed=seed
    missData=data.copy()
    nrow, ncol = missData.shape
    choice_row=random.sample(list(np.linspace(0, nrow-1, nrow, dtype=int)),int(nrow*rate))
    for row in choice_row:
        choice_col=random.sample(list(np.linspace(0, ncol-1, ncol, dtype=int)),int(ncol*rate))
        for col in choice_col:
            missData[row][col]=np.nan
    return missData
def gene_missingdata_chara_bias(rate,data,seed=0):
    random.seed=seed
    missData=data.copy()
    nrow, ncol = missData.shape
    choice_col = random.sample(list(np.linspace(0, ncol - 1, ncol, dtype=int)), int(ncol * rate))
    for col in choice_col:
        choice_row=random.sample(list(np.linspace(0, nrow-1, nrow, dtype=int)),int(nrow*rate))
        for row in choice_row:
            missData[row][col]=np.nan
    return missData
def gene_missingdata_block_bias(rate,data,seed=0):
    random.seed=seed
    missData=data.copy()
    nrow, ncol = missData.shape
    rowlength=random.uniform(rate,1)
    collength=rate/rowlength
    rowlength=round(rowlength*nrow)
    collength=round(collength*ncol)
    start_row=int(random.random()*nrow)
    start_col=int(random.random()*ncol)
    for row in np.arange(0,rowlength):
        for col in np.arange(0,collength):
            missData[(start_row+row)%nrow][(start_col+col)%ncol] = np.nan
    return missData

def gene_missingdata_df(rate,data,seed=0):
    '''
    :param rate: 缺失率，int
    :param data: 完整数据集，np.array
    随机模式缺失
    :return:缺失后数据集和缺失行数
    '''
    random.seed=seed
    missData=data.copy()
    nrow, ncol = missData.shape
    myList = list(np.linspace(0, ncol*nrow-1, ncol*nrow, dtype=int))
    choice_num=random.sample(myList,int(ncol*nrow*rate))
    for num in choice_num:
        row=num%nrow
        col=num//nrow
        missData.iloc[row,col] = np.nan
    return missData
def gene_missingdata_taxa_bias_df(rate,data,seed=0):
    random.seed=seed
    missData=data.copy()
    nrow, ncol = missData.shape
    choice_row=random.sample(list(np.linspace(0, nrow-1, nrow, dtype=int)),int(nrow*rate))
    for row in choice_row:
        choice_col=random.sample(list(np.linspace(0, ncol-1, ncol, dtype=int)),int(ncol*rate))
        for col in choice_col:
            missData.iloc[row,col]=np.nan
    return missData
def gene_missingdata_chara_bias_df(rate,data,seed=0):
    random.seed=seed
    missData=data.copy()
    nrow, ncol = missData.shape
    choice_col = random.sample(list(np.linspace(0, ncol - 1, ncol, dtype=int)), int(ncol * rate))
    for row in choice_col:
        choice_row=random.sample(list(np.linspace(0, nrow-1, nrow, dtype=int)),int(nrow*rate))
        for col in choice_row:
            missData.iloc[row,col]=np.nan
    return missData
def gene_missingdata_block_bias_df(rate,data,seed=0):
    random.seed=seed
    missData=data.copy()
    nrow, ncol = missData.shape
    rowlength=random.uniform(rate,1)
    collength=rate/rowlength
    rowlength=round(rowlength*nrow)
    collength=round(collength*ncol)
    start_row=int(random.random()*nrow)
    start_col=int(random.random()*ncol)
    for row in np.arange(0,rowlength):
        for col in np.arange(0,collength):
            missData.iloc[(start_row+row)%nrow,(start_col+col)%ncol] = np.nan
    return missData
def readDataAndSplit(path):
    csvdata = pd.read_excel(path,header=None)
    dt = csvdata.values
    name=[]
    data = [[] for i in range(len(dt))]
    for i in range(len(dt)):
        for j, val in enumerate(dt[i][1]):
            try:
                data[i].append(int(val))
            except:
                if val == "-":
                    data[i].append(-1)
                elif val == "?":
                    data[i].append(np.nan)
        name.append(dt[i][0])
    misss_row = []
    for ind,i in enumerate(data):
        if np.nan in i:
            misss_row.append(ind)
            continue
    return np.array(data,dtype=float),list(set(misss_row)),name

def check_missrate(data):
    missCount = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            if np.isnan(data[i][j]):
                missCount += 1
    missRate = missCount / (len(data) * len(data[0]))
    return missRate

import heapq
def deleteCharaterBymissRate(originPath,savePath,rate):
    """
    按照缺失率删除特征，特征缺失超过rate进行删除
    :param originPath: 文件原始路径
    :param savePath: 保存路径
    :param rate: 缺失阈值
    :return:
    """
    originData, missRow, speciesName, begin, end = readNex(originPath)
    threshould=int(len(originData[0])*rate)
    missNumber=[]
    for j in range(len(originData[0])):
        count=0
        for i in range(len(originData)):
            if np.isnan(originData[i][j]):
               count+=1
        missNumber.append((count,j))
    ret=heapq.nlargest(threshould,missNumber,key=lambda x:x[0])
    deletRow = [i[1] for i in ret]
    deletRow.sort()
    for index in range(len(begin)):
        pattern="nchar *= *{}".format(str(len(originData[0])))
        substring="nchar = {}".format(str(len(originData[0])-len(deletRow)))
        begin[index]=re.sub(pattern,substring,begin[index])
    for j in deletRow[::-1]:
        originData=np.delete(originData,j,1)

    saveNex(savePath,speciesName, originData, begin, end)

if __name__=="__main__":
    originPat=r"C:\Users\pro\Desktop\实验一缺失模式对建设的影响研究\缺失数据文件"
    path=r'C:\Users\pro\Desktop\实验一缺失模式对建设的影响研究\缺失数据文件\0.75_normal_50taxa100chara_sim.tnt'
    for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
        savePath=r'C:\Users\pro\Desktop\实验一缺失模式对建设的影响研究\dirty\0.75_normal_50taxa100chara_sim_{}.tnt'.format(i)
        deleteCharaterBymissRate(path,savePath,i)
    # i="nchar = 50; "
    # t = re.sub("nchar *= *{}".format(str(50)),
    #            "nchar = {}".format(str(4)),
    #            i)
    # print(t)