#@Time:2019/12/14 11:03
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:handlerMissData.py
__author__ = "skux liu"
import random
import numpy as np
import pandas as pd

def hasNan(li):
    for i in li:
        if np.isnan(i):
            return True
    return False
def geneMissData(rate,data,seed=0):
    '''
    :param rate: 缺失率，int
    :param data: 完整数据集，np.array
    随机模式缺失
    :return:缺失后数据集和缺失行数
    '''
    # random.seed=seed
    count=0
    miss_data=data.copy()
    while count<rate*len(miss_data)*len(miss_data[0]):
        row=int(random.random()*len(miss_data))
        col=int(random.random()*len(miss_data[0]))
        if miss_data[row][col]!=np.nan:
            miss_data[row][col]=np.nan
            count+=1
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
