# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : SOTABaselineMICE.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/21 18:00
"""
import pandas as pd
import numpy as np
import os
from utils.misc_utils import  MAE, masked_mape_np
from ycimpute.utils import evaluate
from ycimpute.imputer import mice

from utils.tools import addResult,plotResult
from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias
from utils.base_tools import modifier
from utils.base_impute import random_inpute
import impyute
from utils.read_file import readNex
from utils.wapper import costTime

@costTime
def imputeMethodMICE(result,originData,missData,missRate,missPattern,dataType='continuous'):
    imputationMethod = "MICE"
    try:

        if dataType!='continuous':
            imputedData = mice.MICE().complete(missData)
            mark = [temp[0] for temp in pd.DataFrame(np.unique(missData)).dropna(axis=0).values]
            imputedData = modifier(imputedData, mark)
        else:
            imputedData = mice.MICE().complete(missData)
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



if __name__=="__main__":
    path = r'../nexus_files'
    for file in os.listdir(path):
        try:
            data, misss_row, speciesname, begin, end = readNex(os.path.join(path, file))
            data = data + 10
        except ValueError:
            print("可能存在数据多态问题")
            #shear_dile(os.path.join(path, file), os.path.join("G:\labWork\cladistic-data-master\可能无用数据"))
            print("文件移动成功")
            continue
        originData =  impyute.imputation.cs.random(data)
        result = {}
        for missPattern in ['normal']:
            for missRate in [0.05]:
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

                result=imputeMethodMICE(result, originData, missData, missRate, missPattern,'disperse')
                break

        plotResult(result)