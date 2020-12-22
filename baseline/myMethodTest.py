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
from utils.tools import saveJson
from logger import logger
from utils.misc_utils import RMSE, MAE, masked_mape_np
from fancyimpute import IterativeImputer
from ycimpute.imputer import mice
from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias

from dnn.tai_test import TAI
from utils.tools import addResult,plotResult
from ycimpute.utils import evaluate
from utils.base_tools import modifier
import datetime
import impyute
from utils.read_file import readNex

def imputeMethod(result,loss,firstImputedMethod,autoMethod,originData,missData,missRate,missPattern,dataType='continuous',firstImputedData="None"):
    isImputed= 'true' if firstImputedData!='None' else 'false'
    imputationMethod =  "{}_{}_{}_{}".format(firstImputedMethod, loss, autoMethod,isImputed)
    try:
        imputedData, firstImputedData = TAI(first_imputation_method=firstImputedMethod,
                                            firstImputedData=firstImputedData,
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


    except Exception as e:
        print(e)
        result = addResult(result, missRate, missPattern, imputationMethod,
                           np.inf,
                           np.inf,
                           np.inf)

    return result
if __name__=="__main__":
    # path = r'../nexus_files'
    # for file in os.listdir(path):
    #     try:
    #         logger.info("**********************{}********************".format(file))
    #         data, misss_row, speciesname, begin, end = readNex(os.path.join(path, file))
    #         data = data + 10
    #     except ValueError:
    #         print("可能存在数据多态问题")
    #         # shear_dile(os.path.join(path, file), os.path.join("G:\labWork\cladistic-data-master\可能无用数据"))
    #         print("文件移动成功")
    #         continue
    #     originData = impyute.imputation.cs.random(data)
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
    
                for firstImputedMethod in ['ii','mice']:
                    # 修改前每次进行预插补
                    for loss in ['MSELoss']:
                        for autoMethod in ['Autoencoder','ResAutoencoder','StockedAutoencoder','StockedResAutoencoder']:
                            start=time.time()
                            result=imputeMethod(result=result,firstImputedMethod=firstImputedMethod,
                                                loss=loss,autoMethod=autoMethod,
                                                originData=originData,missData=missData,
                                                missRate=missRate,missPattern=missPattern)
                            logger.info("改前{}-{}-{}训练耗时:{}".format(firstImputedMethod,loss,autoMethod,time.time() - start))
                    # 修改后，保存第一次预插补结果
                    if firstImputedMethod=='ii':
                        firstImputedData=IterativeImputer().fit_transform(missData)
                    elif firstImputedMethod=='mice':
                        firstImputedData  = mice.MICE().complete(missData)
                    for loss in ['MSELoss']:
                        for autoMethod in ['Autoencoder','ResAutoencoder','StockedAutoencoder','StockedResAutoencoder']:
                            start=time.time()
                            result=imputeMethod(result=result,firstImputedMethod=firstImputedMethod,
                                                firstImputedData=firstImputedData.copy(),
                                                loss=loss,autoMethod=autoMethod,
                                                originData=originData,missData=missData,
                                                missRate=missRate,missPattern=missPattern)
                            logger.info("改后{}-{}-{}训练耗时:{}".format(firstImputedMethod,loss,autoMethod,time.time() - start))
            saveJson(result, "{}_{}_{}_{}.json".format("allMethod", missPattern, file.split("/")[-1],
                                                           datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
        plotResult(result)