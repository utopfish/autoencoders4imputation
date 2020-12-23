# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : pedDataRun.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/22 12:17
"""
import os

import time
import datetime

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

from utils.tools import saveJson
from utils.read_file import readAllTypeFile
from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias

def mainWork(path,savePath):

    for file in os.listdir(path):
        originData = readAllTypeFile(os.path.join(path, file))
        for missPattern in ['normal']:
            # for missPattern in ['normal','block',  'taxa', 'chara']:
            # for i in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
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

                result, _ = Random(result, originData, missData, missRate, missPattern, 'disperse')
                result, _ = Medain(result, originData, missData, missRate, missPattern, 'disperse')
                result, KNNImputedData = KNN(result, originData, missData, missRate, missPattern, 'disperse')
                result, EMImputedData = EM(result, originData, missData, missRate, missPattern, 'disperse')
                result, IIImputedData = II(result, originData, missData, missRate, missPattern, 'disperse')
                result, _ = GAIN(result, originData, missData, missRate, missPattern, 'disperse')
                result, _ = MIDA(result, originData, missData, missRate, missPattern, 'disperse')
                result, MICEImputedData = MICE(result, originData, missData, missRate, missPattern, 'disperse')
                # for firstImputedMethod in ['ii', 'mice']:
                for firstImputedMethod in ['knn', 'ii', 'mice']:
                    if firstImputedMethod == 'knn':
                        firstImputedData = KNNImputedData
                    elif firstImputedMethod == 'ii':
                        firstImputedData = IIImputedData
                    elif firstImputedMethod == 'mice':
                        firstImputedData = MICEImputedData
                    for loss in ['MSELoss']:
                        # for autoMethod in ['Autoencoder','ResAutoencoder','StockedAutoencoder','StockedResAutoencoder']:
                        for autoMethod in ['Autoencoder']:
                            start = time.time()
                            result = TAI(result=result, firstImputedMethod=firstImputedMethod,
                                         firstImputedData=firstImputedData.copy(),
                                         loss=loss, autoMethod=autoMethod,
                                         originData=originData, missData=missData,
                                         missRate=missRate, missPattern=missPattern)
                            logger.info(
                                "改后{}-{}-{}训练耗时:{}".format(firstImputedMethod, loss, autoMethod, time.time() - start))
            saveJson(result,
                     os.path.join(savePath,"{}_{}_{}_{}.json".format("allmethod", missPattern, file,
                                               datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))))
            # plotResult(result)
if __name__=="__main__":
    path = r'../nexus_files'
    savePath = r'../experiment/publicData'
    for _ in range(3):
        mainWork(path,savePath)
