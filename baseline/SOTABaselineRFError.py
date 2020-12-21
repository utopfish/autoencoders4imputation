# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : SOTABaselineRFError.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/21 15:41
"""
from ycimpute.imputer import MissForest
import pandas as pd
import numpy as np
## TODO:在离散值和连续值的处理上，这个还有bug，看之后能否进行处理
from utils.misc_utils import RMSE, MAE, masked_mape_np
from logger import logger
from ycimpute.utils import evaluate
file = r'../public_data/1_Iris.xlsx'
data = pd.read_excel(file, sheet_name="dataset")
dt = np.array(data.values)
data = dt.astype('float')
originData = data[:-1]
target = data[-1]
result=[]
from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias
for missPattern in ['normal']:
    for i in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if missPattern == 'normal':
            missData = gene_missingdata(rate=i, data=originData)
        elif missPattern == 'taxa':
            missData = gene_missingdata_taxa_bias(rate=i, data=originData)
        elif missPattern == 'chara':
            missData = gene_missingdata_chara_bias(rate=i, data=originData)
        elif missPattern == 'block':
            missData = gene_missingdata_block_bias(rate=i, data=originData)
        else:
            raise Exception("缺失模式错误，请在'normal','taxa','chara','block'中选择对应模式")
        tmp = [[] for _ in range(3)]
        # try:
        method="RF"
        tmpData=missData.astype('float')
        imputedData = MissForest().complete(missData)

        score = evaluate.RMSE(originData, imputedData)
        tmp[0].append(score)
        tmp[1].append(MAE(originData, imputedData))
        tmp[2].append(masked_mape_np(originData, imputedData))
        logger.info("{} missing rate:{},RMSE:{}".format(method,i, score))
        # except Exception as e:
        #     print(e)
        #
        #     tmp[0].append(np.inf)
        #     tmp[1].append(np.inf)
        #     tmp[2].append(np.inf)
        result.append({"missRate":i,
                       "missPattern":missPattern,
                       "imputationMethod":method,
                       "RMSE":tmp[0],
                       "MAE":tmp[1],
                       "masked_mape_np":tmp[2]})
        print(result)


