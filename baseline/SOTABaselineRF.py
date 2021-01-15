# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : SOTABaselineRF.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/21 15:41
"""
from ycimpute.imputer import MissForest
import pandas as pd
import numpy as np
## TODO:在离散值和连续值的处理上，这个还有bug，看之后能否进行处理
from utils.misc_utils import RMSE, MAE, masked_mape_np
from logger import logger
from ycimpute.utils import evaluate
from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias
from utils.tools import addResult,plotResult,saveJson
from utils.base_tools import modifier
from predictive_imputer import predictive_imputer
from utils.wapper import costTime
@costTime
def imputeMethodMR(result,originData,missData,missRate,missPattern,dataType='continuous'):
    imputationMethod = "RandomForest"
    try:
        imputer = predictive_imputer.PredictiveImputer(f_model='RandomForest')
        imputedData = imputer.fit(missData).transform(missData.copy())
        if dataType!='continuous':
            mark = [temp[0] for temp in pd.DataFrame(np.unique(missData)).dropna(axis=0).values]
            imputedData = modifier(imputedData, mark)
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
    file = r'../public_data/1_Iris.xlsx'
    resultPath=r'../result'
    data = pd.read_excel(file, sheet_name="dataset")
    dt = np.array(data.values)
    data = dt.astype('float')
    originData = data[:-1]
    target = data[-1]
    result = {}
    for missPattern in ['normal']:
        for missRate in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
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

            result,_=imputeMethodMR(result, originData, missData, missRate, missPattern)
    #saveJson(result,"{}_{}_{}.json".format("EM","iris",datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))