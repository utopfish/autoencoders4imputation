# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : SOTABaselineEM.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/21 16:36
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
from utils.base_tools import modifier
from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias
from utils.wapper import costTime

@costTime
def imputeMethodEM(result,originData,missData,missRate,missPattern,dataType='continuous'):
    imputationMethod = "EM"
    try:
        imputedData = EM().complete(missData)
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


