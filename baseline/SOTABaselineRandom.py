# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : SOTABaselineRandom.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/21 19:51
"""
import datetime
from ycimpute.imputer import EM
import pandas as pd
import numpy as np
from utils.tools import addResult,plotResult,saveJson
from utils.handler_loss import MSE
from utils.misc_utils import RMSE, MAE, masked_mape_np
from logger import logger
from ycimpute.utils import evaluate
import impyute
from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias
from utils.base_tools import modifier

from utils.wapper import costTime

@costTime
def imputeMethodRandom(result,originData,missData,missRate,missPattern,dataType='continuous'):
    imputationMethod = "Random"
    try:
        imputedData = impyute.imputation.cs.random(missData)
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
    resultPath=r'../result'
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
    #saveJson(result,"{}_{}_{}.json".format("EM","iris",datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    plotResult(result)