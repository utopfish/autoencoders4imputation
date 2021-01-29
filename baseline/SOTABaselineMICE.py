# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : SOTABaselineMICE.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/21 18:00
"""
import pandas as pd
import numpy as np
import os
from utils.misc_utils import  MAE, masked_mape_np
from ycimpute.utils import evaluate
from ycimpute.imputer import mice

from utils.tools import addResult,plotResult
from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias
from utils.base_tools import modifier
import impyute
from utils.read_file import readNex,lableEncoder
from utils.wapper import costTime

@costTime
def imputeMethodMICE(result,originData,missData,missRate,missPattern,dataType='continuous'):
    imputationMethod = "MICE"
    try:

        if dataType!='continuous':
            imputedData = mice.MICE().complete(missData)
            mark = [temp[0] for temp in pd.DataFrame(np.unique(missData)).dropna(axis=0).values]
            imputedData = modifier(imputedData, mark)
        else:
            imputedData = mice.MICE().complete(missData)
        result = addResult(result, missRate, missPattern, imputationMethod,
                           evaluate.RMSE(originData, imputedData),
                           MAE(originData, imputedData),
                           masked_mape_np(originData, imputedData))
    except Exception as e:
        print(e)
        imputedData = 'none'
        result = addResult(result, missRate, missPattern, imputationMethod,
                           np.inf,
                           np.inf,
                           np.inf)
    return result,imputedData



if __name__=="__main__":
    dataPath = r'C:\Users\pro\Desktop\实验二自编码器建树\古生物数据集测试\01起始数据集\01_Yang2015.nex'

    missData, missRow, speciesName, begin, end = readNex(dataPath)
    #missData = lableEncoder(originData)
    result={}
    missRate=0.3
    missPattern="normal"
    print(missData)
    s=set()
    for i in range(len(missData)):
        for j in range(len(missData[0])):
            s.add(missData[i][j])
    print(s)
    print(np.isnan(missData).any())
    print(np.isfinite(missData).all())
    t=mice.MICE().complete(missData)