# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : baseline_ped.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/21 21:50
"""
import os

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
from utils.read_file import readNex
from baseline.myMethodTest import imputeMethod as TAI

from utils.tools import addResult,plotResult,saveJson

from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias
from utils.base_impute import random_inpute
import impyute
if __name__=="__main__":

    path = r'../nexus_files'
    pciturePath = r'../result/picture'
    save_path=r'../result/ped'
    total_result_half={}
    total_result_all={}
    for file in os.listdir(path):
        try:
            logger.info("**********************{}********************".format(file))
            data, misss_row, speciesname, begin, end = readNex(os.path.join(path, file))
            data = data + 10
        except ValueError:
            print("可能存在数据多态问题")
            #shear_dile(os.path.join(path, file), os.path.join("G:\labWork\cladistic-data-master\可能无用数据"))
            print("文件移动成功")
            continue
        originData =  impyute.imputation.cs.random(data)
        for missPattern in ['normal']:
        #for missPattern in ['normal','block',  'taxa', 'chara']:
            # for i in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for missRate in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
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
                result={}
                result = Random(result, originData, missData, missRate, missPattern,'disperse')
                result = Medain(result, originData, missData, missRate, missPattern,'disperse')
                result = KNN(result, originData, missData, missRate, missPattern,'disperse')
                result = EM(result, originData, missData, missRate, missPattern,'disperse')
                result = II(result, originData, missData, missRate, missPattern,'disperse')
                result = GAIN(result, originData, missData, missRate, missPattern,'disperse')
                result = MIDA(result, originData, missData, missRate, missPattern,'disperse')
                result = MICE(result, originData, missData, missRate, missPattern,'disperse')
                for firstImputedMethod in ['ii', 'mice']:
                    for loss in ['MSELoss']:
                        for autoMethod in ['Autoencoder', 'ResAutoencoder', 'StockedAutoencoder',
                                           'StockedResAutoencoder']:
                            start = time.time()
                            result = TAI(result=result, firstImputedMethod=firstImputedMethod,
                                         loss=loss, autoMethod=autoMethod,
                                         originData=originData, missData=missData,
                                         missRate=missRate, missPattern=missPattern,dataType='disperse')

                            logger.info("训练耗时:{}".format(time.time() - start))
            saveJson(result,
                     "{}_{}_{}_{}.json".format("allmethod",missPattern, file.split("/")[-1], datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
            plotResult(result)
