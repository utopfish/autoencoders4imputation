# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : dataAnalysis.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/21 20:47
"""

from prettytable import PrettyTable
import json
import numpy as np
from logger import logger
def readJson(fileName):
    with open("../result/{}".format(fileName),'r') as f:
        jsonData=json.load(f)
    return jsonData

if __name__=="__main__":
    result=readJson("all_1_Iris.xlsx_20201221-210256.json")
    missRate = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    x = PrettyTable(['Method', 'pattern', 'missRate', 'RMSE', 'MAE', "MAPE"])
    rank = [[] for _ in range(len(missRate))]
    methods=[]
    for patternMethod in result.keys():
        pattern = patternMethod.split(":")[0]
        method = patternMethod.split(":")[1]
        methods.append(method)
        for index,row in enumerate(result[patternMethod]):
            rank[index].append(row)
            try:
                temp = ["{:.4f}".format(i) for i in row]
            except:
                temp = ['NaN' for _ in row]
            x.add_row([method] + [pattern] + ["{:.2f}".format(missRate[index])] + temp)
    rank=np.array(rank)
    for index,missRateData in enumerate(rank):
        pos = np.where(missRateData==np.min(missRateData,axis=0))
        tmp=[methods[miscIndex] for miscIndex in list(pos[0])]
        print("缺失率为{}，下RMSE，MAP，MAPE最小值对应方法".format(missRate[index]),tmp)
    top_k = 3
    for index, missRateData in enumerate(rank):

        arr = missRateData[:,0]
        top_k_idx = arr.argsort()[0:top_k]
        tmp = [methods[miscIndex] for miscIndex in top_k_idx]
        print("缺失率为{}下,RMSE前三对应方法".format(missRate[index]), tmp)
        arr = missRateData[:,1]
        top_k_idx = arr.argsort()[0:top_k]
        tmp = [methods[miscIndex] for miscIndex in top_k_idx]
        print("缺失率为{}下,MAP前三对应方法".format(missRate[index]), tmp)
        arr = missRateData[:,2]
        top_k_idx = arr.argsort()[0:top_k]
        tmp = [methods[miscIndex] for miscIndex in top_k_idx]
        print("缺失率为{}下,MAPE前三对应方法".format(missRate[index]), tmp)
    # logger.warning(x)

