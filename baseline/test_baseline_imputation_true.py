# """
# @author: liuAmon
# @contact:utopfish@163.com
# @file: test_baseline_imputation_true.py
# @time: 2020/7/22 20:04
# """

## 比较在UCI数据集上MICE与AE的区别


from baseline.myMethod import imputeMethod as TAI

import time
import datetime

from logger import logger


from baseline.SOTABaselineMICE import  imputeMethodMICE as MICE

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

                    result, MICEImputedData = MICE(result, originData, missData, missRate, missPattern)
                    for firstImputedMethod in ['mice']:
                        if firstImputedMethod=='mice':
                            firstImputedData  = MICEImputedData
                        for loss in ['MSELoss']:
                            #for autoMethod in ['Autoencoder','ResAutoencoder','StockedAutoencoder','StockedResAutoencoder']:
                            for autoMethod in ['StockedResAutoencoder']:
                                start=time.time()
                                result,_=TAI(result=result,firstImputedMethod=firstImputedMethod,
                                                    firstImputedData=firstImputedData,
                                                    loss=loss,autoMethod=autoMethod,
                                                    originData=originData,missData=missData,
                                                    missRate=missRate,missPattern=missPattern,
                                             )
                                logger.info("{}-{}-{}训练耗时:{}".format(firstImputedMethod,loss,autoMethod,time.time() - start))
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                saveJson(result, os.path.join(savePath,"{}_{}_{}_{}.json".format("allMethod", missPattern,file, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))))
                # plotResult(result)

if __name__=="__main__":
    path = r'../public_data'
    savePath=r'E:\labCode\autoencoders4imputation\experiment\加大自编码器参数-mice'
    for _ in range(2):
        mainWork(path,savePath)

