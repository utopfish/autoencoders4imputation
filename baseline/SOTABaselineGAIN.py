# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : SOTABaselineGAIN.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/21 17:24
"""

import pandas as pd
import numpy as np

from utils.misc_utils import  MAE, masked_mape_np
from logger import logger
from ycimpute.utils import evaluate
from dnn.gain import GAIN

from utils.tools import addResult,plotResult
from utils.base_tools import modifier
from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias




def imputeMethodGAIN(result,originData,missData,missRate,missPattern,dataType='continuous'):
    imputationMethod = "GAIN"
    try:
        imputedData = GAIN().complete(missData)
        if dataType!='continuous':

            mark = [temp[0] for temp in pd.DataFrame(np.unique(missData)).dropna(axis=0).values]
            imputedData = modifier(imputedData, mark)
        result = addResult(result, missRate, missPattern, imputationMethod,
                           evaluate.RMSE(originData, imputedData),
                           MAE(originData, imputedData),
                           masked_mape_np(originData, imputedData))
    except Exception as e:
        print(e)
        result = addResult(result, missRate, missPattern, imputationMethod,
                           np.inf,
                           np.inf,
                           np.inf)
    return result



if __name__=="__main__":
    file = r'../public_data/1_Iris.xlsx'
    data = pd.read_excel(file, sheet_name="dataset")
    dt = np.array(data.values)
    data = dt.astype('float')
    originData = data[:-1]
    target = data[-1]
    result = {}
    for missPattern in ['normal']:
        for missRate in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            if missPattern == 'normal':
                missData = gene_missingdata(rate=missRate, data=originData)
            elif missPattern == 'taxa':
                missData = gene_missingdata_taxa_bias(rate=missRate, data=originData)
            elif missPattern == 'chara':
                missData = gene_missingdata_chara_bias(rate=missRate, data=originData)
            elif missPattern == 'block':
                missData = gene_missingdata_block_bias(rate=missRate, data=originData)
            else:
                raise Exception("缺失模式错误，请在'normal','taxa','chara','block'中选择对应模式")

            result=imputeMethod(result, originData, missData, missRate, missPattern)

    plotResult(result)