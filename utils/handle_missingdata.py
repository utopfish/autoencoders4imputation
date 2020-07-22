#@Time:2019/12/14 11:03
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:handle_missingdata.py
__author__ = "skux liu"
import random
import numpy as np
import pandas as pd

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
    random.seed=seed
    miss_data=data.copy()
    nrow, ncol = miss_data.shape
    myList = list(np.linspace(0, ncol*nrow-1, ncol*nrow, dtype=int))
    choice_num=random.sample(myList,int(ncol*nrow*rate))
    for num in choice_num:
        row=num%nrow
        col=num//nrow
        miss_data[row][col] = np.nan
    return miss_data
def gene_missingdata_taxa_bias(rate,data,seed=0):
    random.seed=seed
    miss_data=data.copy()
    nrow, ncol = miss_data.shape
    choice_row=random.sample(list(np.linspace(0, nrow-1, nrow, dtype=int)),int(nrow*rate))
    for row in choice_row:
        choice_col=random.sample(list(np.linspace(0, ncol-1, ncol, dtype=int)),int(ncol*rate))
        for col in choice_col:
            miss_data[row][col]=np.nan
    return miss_data
def gene_missingdata_chara_bias(rate,data,seed=0):
    random.seed=seed
    miss_data=data.copy()
    nrow, ncol = miss_data.shape
    choice_col = random.sample(list(np.linspace(0, ncol - 1, ncol, dtype=int)), int(ncol * rate))
    for col in choice_col:
        choice_row=random.sample(list(np.linspace(0, nrow-1, nrow, dtype=int)),int(nrow*rate))
        for row in choice_row:
            miss_data[row][col]=np.nan
    return miss_data
def gene_missingdata_block_bias(rate,data,seed=0):
    random.seed=seed
    miss_data=data.copy()
    nrow, ncol = miss_data.shape
    rowlength=random.uniform(rate,1)
    collength=rate/rowlength
    rowlength=round(rowlength*nrow)
    collength=round(collength*ncol)
    start_row=int(random.random()*nrow)
    start_col=int(random.random()*ncol)
    for row in np.arange(0,rowlength):
        for col in np.arange(0,collength):
            miss_data[(start_row+row)%nrow][(start_col+col)%ncol] = np.nan
    return miss_data

def gene_missingdata_df(rate,data,seed=0):
    '''
    :param rate: 缺失率，int
    :param data: 完整数据集，np.array
    随机模式缺失
    :return:缺失后数据集和缺失行数
    '''
    random.seed=seed
    miss_data=data.copy()
    nrow, ncol = miss_data.shape
    myList = list(np.linspace(0, ncol*nrow-1, ncol*nrow, dtype=int))
    choice_num=random.sample(myList,int(ncol*nrow*rate))
    for num in choice_num:
        row=num%nrow
        col=num//nrow
        miss_data.iloc[row,col] = np.nan
    return miss_data
def gene_missingdata_taxa_bias_df(rate,data,seed=0):
    random.seed=seed
    miss_data=data.copy()
    nrow, ncol = miss_data.shape
    choice_row=random.sample(list(np.linspace(0, nrow-1, nrow, dtype=int)),int(nrow*rate))
    for row in choice_row:
        choice_col=random.sample(list(np.linspace(0, ncol-1, ncol, dtype=int)),int(ncol*rate))
        for col in choice_col:
            miss_data.iloc[row,col]=np.nan
    return miss_data
def gene_missingdata_chara_bias_df(rate,data,seed=0):
    random.seed=seed
    miss_data=data.copy()
    nrow, ncol = miss_data.shape
    choice_col = random.sample(list(np.linspace(0, ncol - 1, ncol, dtype=int)), int(ncol * rate))
    for row in choice_col:
        choice_row=random.sample(list(np.linspace(0, nrow-1, nrow, dtype=int)),int(nrow*rate))
        for col in choice_row:
            miss_data.iloc[row,col]=np.nan
    return miss_data
def gene_missingdata_block_bias_df(rate,data,seed=0):
    random.seed=seed
    miss_data=data.copy()
    nrow, ncol = miss_data.shape
    rowlength=random.uniform(rate,1)
    collength=rate/rowlength
    rowlength=round(rowlength*nrow)
    collength=round(collength*ncol)
    start_row=int(random.random()*nrow)
    start_col=int(random.random()*ncol)
    for row in np.arange(0,rowlength):
        for col in np.arange(0,collength):
            miss_data.iloc[(start_row+row)%nrow,(start_col+col)%ncol] = np.nan
    return miss_data
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