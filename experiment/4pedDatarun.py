# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : 4pedDatarun.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2021/1/13 18:59
"""

import os
import impyute
from utils.read_file  import readNex,saveData,lableEncoder

from logger import logger
import time
import datetime
import tqdm
from utils.tools import saveJson
from baseline.SOTABaselineKNN import  imputeMethodKNN as KNN
from baseline.SOTABaselineMICE import  imputeMethodMICE as MICE
from baseline.SOTABaselineRandom import  imputeMethodRandom as Random
from baseline.SOTABaselineEM import  imputeMethodEM as EM
from baseline.SOTABaselineMedian import  imputeMethodMedain as Medain
from baseline.myMethodTest import imputeMethod as TAI

from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias
originDataPath=r'C:\Users\pro\Desktop\实验相关文档\缺失插补建树全流程\01起始含缺失数据'
noMissingDataPath=r'C:\Users\pro\Desktop\实验相关文档\缺失插补建树全流程\02随机插补无缺失数据'
simDataPath=r'C:\Users\pro\Desktop\实验相关文档\缺失插补建树全流程\03无缺失随机缺失模拟数据'
#生成模拟数据集
# for file in os.listdir(originDataPath):
#     data,missRow,speciesName,begin,end=readNex(os.path.join(originDataPath,file))
#     noMissingData = impyute.imputation.cs.random(data)
#     saveData(noMissingDataPath,file,speciesName,noMissingData,begin,end)

imputedDataPath = r'C:\Users\pro\Desktop\实验相关文档\缺失插补建树全流程\04缺失插补结果'
for file in tqdm.tqdm(os.listdir(noMissingDataPath)):
    originData, missRow, speciesName, begin, end = readNex(os.path.join(noMissingDataPath, file))
    for missPattern in ['normal']:
        result = {}
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
            saveData(simDataPath, "{}_{}".format(str(missRate),file), speciesName, missData, begin, end)
            missData = lableEncoder(missData)
            result, RandomImputedData = Random(result, originData, missData, missRate, missPattern, 'disperse')
            saveData(os.path.join(imputedDataPath, "Random"), "{}_{}".format(str(missRate),file), speciesName,
                     RandomImputedData, begin, end)

            result, MedainImputedData = Medain(result, originData, missData, missRate, missPattern)
            saveData(os.path.join(imputedDataPath, "Medain"), "{}_{}".format(str(missRate),file), speciesName,
                     MedainImputedData, begin, end)
            result, EMImputedData = EM(result, originData, missData, missRate, missPattern, 'disperse')
            saveData(os.path.join(imputedDataPath, "EM"), "{}_{}".format(str(missRate),file), speciesName,
                     EMImputedData, begin, end)
            result, KNNImputedData = KNN(result, originData, missData, missRate, missPattern, 'disperse')
            saveData(os.path.join(imputedDataPath, "KNN"), "{}_{}".format(str(missRate),file), speciesName,
                     KNNImputedData, begin, end)

            start = time.time()
            #result, MICEImputedData = MICE(result, originData, missData, missRate, missPattern, 'disperse')
            firstImputedData = KNNImputedData
            firstImputedMethod='knn'
            for loss in ['MSELoss']:
                for autoMethod in ['Autoencoder']:
                    result, TAIimputedData = TAI(result=result, firstImputedMethod=firstImputedMethod,
                                                 firstImputedData=firstImputedData,
                                                 loss=loss, autoMethod=autoMethod,
                                                 originData=originData, missData=missData,
                                                 missRate=missRate, missPattern=missPattern,
                                                 dataType='disperse')
                    logger.info(
                        "改后{}-{}-{}训练耗时:{}".format(firstImputedMethod, loss, autoMethod, time.time() - start))
                    saveData(os.path.join(imputedDataPath, "ours"), "{}_{}".format(str(missRate),file), speciesName,
                             TAIimputedData, begin, end)
    saveJson(result,
             os.path.join(imputedDataPath, "{}_{}_{}_{}.json".format("allmethod", missPattern, file,
                                                                     datetime.datetime.now().strftime(
                                                                         '%Y%m%d-%H%M%S'))))