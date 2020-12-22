# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : baseline_isir.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/21 17:50
"""
import time
import datetime
import pandas as pd
import numpy as np
from logger import logger
from baseline.SOTABaselineGAIN import  imputeMethodGAIN as  GAIN
from baseline.SOTABaselineII import  imputeMethodII as II
from baseline.SOTABaselineKNN import  imputeMethodKNN as KNN
from baseline.SOTABaselineMedian import  imputeMethodMedain as Medain
from baseline.SOTABaselineMIDA import  imputeMethodMIDA as MIDA
from baseline.SOTABaselineEM import  imputeMethodEM as EM
from baseline.SOTABaselineMICE import  imputeMethodMICE as MICE
from baseline.SOTABaselineRandom import  imputeMethodRandom as Random

from baseline.myMethodTest import imputeMethod as TAI

from utils.tools import addResult,plotResult,saveJson

from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias
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

            result = Random(result, originData, missData, missRate, missPattern)
            result = Medain(result, originData, missData, missRate, missPattern)
            result = KNN(result, originData, missData, missRate, missPattern)
            result = EM(result, originData, missData, missRate, missPattern)
            result = II(result, originData, missData, missRate, missPattern)
            result = GAIN(result, originData, missData, missRate, missPattern)
            result = MIDA(result, originData, missData, missRate, missPattern)
            result = MICE(result, originData, missData, missRate, missPattern)
            for firstImputedMethod in ['ii', 'mice']:
                for loss in ['MSELoss']:
                    for autoMethod in ['Autoencoder', 'ResAutoencoder', 'StockedAutoencoder', 'StockedResAutoencoder']:
                        start = time.time()
                        result = TAI(result=result, firstImputedMethod=firstImputedMethod,
                                              loss=loss, autoMethod=autoMethod,
                                              originData=originData, missData=missData,
                                              missRate=missRate, missPattern=missPattern)

                        logger.info("训练耗时:{}".format(time.time() - start))
    saveJson(result, "{}_{}_{}.json".format("all", file.split("/")[-1], datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    plotResult(result)