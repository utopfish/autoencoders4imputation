# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : myMethod.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/21 19:57
"""
import numpy as np
import pandas as pd
from utils.misc_utils import RMSE, MAE, masked_mape_np


from dnn.tai_test import TAI
from utils.tools import addResult,plotResult
from ycimpute.utils import evaluate
from utils.base_tools import modifier
from utils.dataAnalysis import analysisMeanMain,json2Excel
import datetime
def imputeMethod(result,loss,firstImputedMethod,autoMethod,originData,missData,missRate,missPattern,dataType='continuous',firstImputedData="None"):
    imputationMethod =  "{}_{}_{}".format(firstImputedMethod, loss, autoMethod)
    try:
        imputedData, firstImputedData = TAI(first_imputation_method=firstImputedMethod,
                                            firstImputedData=firstImputedData,
                                               batch_size=len(missData),
                                               epochs=500,
                                               theta=int(len(missData[0]) / 3),
                                               iterations=100,
                                               Autoencoder_method=autoMethod,
                                               loss=loss,
                                               use_cuda=False
                                               ).complete(missData)
        
        if dataType!='continuous':
            mark = [temp[0] for temp in pd.DataFrame(np.unique(missData)).dropna(axis=0).values]
            imputedData = modifier(imputedData, mark)

        result = addResult(result, missRate, missPattern, imputationMethod,
                           evaluate.RMSE(originData, imputedData),
                           MAE(originData, imputedData),
                           masked_mape_np(originData, imputedData))

        return result, imputedData
    except Exception as e:
        print(e)
        result = addResult(result, missRate, missPattern, imputationMethod,
                           np.inf,
                           np.inf,
                           np.inf)

        return result,firstImputedData
if __name__=="__main__":
    dataPath = r'E:\labCode\autoencoders4imputation\baseline'
    savePath = r'C:\Users\pro\Desktop\自编码器鸢尾花结果验证'
    json2Excel(dataPath, savePath)
