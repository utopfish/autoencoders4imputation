# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : myMethodTest.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/21 19:57
"""
import os
import time
import numpy as np
import pandas as pd

from logger import logger
from utils.misc_utils import RMSE, MAE, masked_mape_np

from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias

from dnn.tai_test import TAI
from utils.tools import addResult,plotResult
from ycimpute.utils import evaluate
from utils.base_tools import modifier

def imputeMethod(result,loss,firstImputedMethod,autoMethod,originData,missData,missRate,missPattern,dataType='continuous'):
    imputationMethod =  "{}_{}_{}".format(firstImputedMethod, loss, autoMethod)
    try:
        imputedData, firstImputedData = TAI(first_imputation_method=firstImputedMethod,
                                               batch_size=len(missData),
                                               epochs=300,
                                               theta=int(len(missData[0]) / 2),
                                               iterations=30,
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
        # name = "{}:{}".format(missPattern, firstImputedMethod)
        # if name not in result.keys():
        #     result = addResult(result, missRate, missPattern, firstImputedMethod,
        #                    evaluate.RMSE(originData, firstImputedData),
        #                    MAE(originData, imputedData),
        #                    masked_mape_np(originData, firstImputedData))
        # logger.info(
        #     "{}_{}_{}_{}_{} first missing rate:{},RMSE:{}".format(file, missPattern, firstImputedMethod, loss, method, missRate,
        #                                                           score1))
        # logger.info(
        #     "{}_{}_{}_{}_{} missing rate:{},RMSE:{}".format(file, missPattern, firstImputedMethod, loss, method, missRate,
        #                                                     score))

    except Exception as e:
        print(e)
        result = addResult(result, missRate, missPattern, imputationMethod,
                           np.inf,
                           np.inf,
                           np.inf)
        # name = "{}:{}".format(missPattern, firstImputedMethod)
        # if name not in result.keys():
        #     result = addResult(result, missRate, missPattern, firstImputedMethod,
        #                    np.inf,
        #                    np.inf,
        #                    np.inf)
    return result
if __name__=="__main__":
    path = r'../public_data/'
    pciturePath = r'../result/picture'
    save_path=r'../result/pub'
    total_result_half={}
    total_result_all={}
    for file in os.listdir(path):
        result={}
        file='1_Iris.xlsx'
        logger.info("**********************{}********************".format(file))
        data = pd.read_excel(os.path.join(path, file), sheet_name="dataset")
        dt = np.array(data.values)
        data = dt.astype('float')
        originData = data[:, :-1]
        target = data[:, -1]
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
    
                for firstImputedMethod in ['ii','mice']:
                    for loss in ['MSELoss']:
                        for autoMethod in ['Autoencoder','ResAutoencoder','StockedAutoencoder','StockedResAutoencoder']:
                            
                            start=time.time()
                            result=imputeMethod(result=result,firstImputedMethod=firstImputedMethod,
                                                loss=loss,autoMethod=autoMethod,
                                                originData=originData,missData=missData,
                                                missRate=missRate,missPattern=missPattern)

                            logger.info("训练耗时:{}".format(time.time() - start))

        plotResult(result)
        break