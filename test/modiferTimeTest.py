# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : modiferTimeTest.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/21 22:14
"""
## 测试modifer 函数对时间的影响
import os

import time
import datetime
import pandas as pd
import numpy as np
from logger import logger
from baseline.SOTABaselineGAIN import imputeMethod as GAIN
from baseline.SOTABaselineII import imputeMethod as II
from baseline.SOTABaselineKNN import imputeMethod as KNN
from baseline.SOTABaselineMedian import imputeMethod as Medain
from baseline.SOTABaselineMIDA import imputeMethod as MIDA
from baseline.SOTABaselineEM import imputeMethod as EM
from baseline.SOTABaselineMICE import imputeMethod as MICE
from baseline.SOTABaselineRandom import imputeMethod as Random
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
        originData = imputedData = impyute.imputation.cs.random(data)
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
                start = time.time()
                result = Random(result, originData, missData, missRate, missPattern)
                logger.info("不加modifer耗时:{}".format(time.time() - start))

                start = time.time()
                result = Random(result, originData, missData, missRate, missPattern, 'disperse')
                logger.info("加modifer耗时:{}".format(time.time() - start))
        # saveJson(result,
        #          "{}_{}_{}.json".format("all", file.split("/")[-1], datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
        plotResult(result)
        break