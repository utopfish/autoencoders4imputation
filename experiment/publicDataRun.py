# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : publicDataRun.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/21 22:53
"""

import time
import datetime

from logger import logger
from fancyimpute import IterativeImputer
from ycimpute.imputer import mice
from baseline.SOTABaselineGAIN import  imputeMethodGAIN as  GAIN
from baseline.SOTABaselineII import  imputeMethodII as II
from baseline.SOTABaselineKNN import  imputeMethodKNN as KNN
from baseline.SOTABaselineMedian import  imputeMethodMedain as Medain
from baseline.SOTABaselineMIDA import  imputeMethodMIDA as MIDA
from baseline.SOTABaselineEM import  imputeMethodEM as EM
from baseline.SOTABaselineMICE import  imputeMethodMICE as MICE
from baseline.SOTABaselineRandom import  imputeMethodRandom as Random
from baseline.SOTABaselineRF import imputeMethodMR as MR
from baseline.myMethod import imputeMethod as TAI

from utils.tools import saveJson
from utils.read_file import readAllTypeFile
from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias
import os
import tqdm

def mainWork(path,savePath):
    pbar = tqdm.tqdm(os.listdir(path), desc='dirs')
    for file in pbar:
        pbar.set_description("Processing %s" % file)
        if file.endswith('xlsx') or file.endswith('csv'):
            originData = readAllTypeFile(os.path.join(path, file))
            for missPattern in ['normal']:
            # for missPattern in ['normal','block',  'taxa', 'chara']:
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

                    result,_ = Random(result, originData, missData, missRate, missPattern)
                    result,_ = Medain(result, originData, missData, missRate, missPattern)
                    result,KNNImputedData = KNN(result, originData, missData, missRate, missPattern)
                    result,_ = EM(result, originData, missData, missRate, missPattern)
                    result,IImputedData = II(result, originData, missData, missRate, missPattern)
                    result,_ = GAIN(result, originData, missData, missRate, missPattern)
                    result, MRImputedData = MR(result, originData, missData, missRate, missPattern)
                    result,_ = MIDA(result, originData, missData, missRate, missPattern)
                    result, MICEImputedData = MICE(result, originData, missData, missRate, missPattern)
                    for firstImputedMethod in ['mice']:
                        if firstImputedMethod=='ii':
                            firstImputedData=IImputedData
                        elif firstImputedMethod=='mice':
                            firstImputedData  = MICEImputedData
                        elif firstImputedMethod =='knn':
                            firstImputedData=KNNImputedData
                        for loss in ['MSELoss']:
                            #for autoMethod in ['Autoencoder','ResAutoencoder','StockedAutoencoder','StockedResAutoencoder']:
                            for autoMethod in ['StockedResAutoencoder']:
                                start=time.time()
                                result,_=TAI(result=result,firstImputedMethod=firstImputedMethod,
                                                    firstImputedData=firstImputedData,
                                                    loss=loss,autoMethod=autoMethod,
                                                    originData=originData,missData=missData,
                                                    missRate=missRate,missPattern=missPattern)
                                logger.info("{}-{}-{}训练耗时:{}".format(firstImputedMethod,loss,autoMethod,time.time() - start))
                saveJson(result, os.path.join(savePath,"{}_{}_{}_{}.json".format("allMethod", missPattern,file, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))))
                # plotResult(result)

if __name__=="__main__":
    path = r'../public_data'
    savePath=r'E:\labCode\autoencoders4imputation\experiment\加大自编码器参数'
    for _ in range(5):
        mainWork(path,savePath)

