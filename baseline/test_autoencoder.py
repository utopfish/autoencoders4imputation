#@Time:2020/8/7 19:43
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:test_autoencoder.py
__author__ = "liuAmon"

import os
import time
import impyute
import numpy as np
import pandas as pd

from utils.base_tools import merge_two_dicts
from prettytable import PrettyTable
from logger import logger

from utils.misc_utils import RMSE, MAE, masked_mape_np
from matplotlib import pyplot as plt
from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias
from dnn.mida import MIDA
from dnn.gain import GAIN
from dnn.tai_test import TAI
from ycimpute.imputer import knnimput, mice, EM
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler, SimpleFill
from sklearn.datasets import load_boston
from utils.read_file import write_excel_xls_append,write_excel_xls
path = r'../public_data/'
pciturePath = r'../result/picture'
save_path=r'../result/pub'
total_result_half={}
total_result_all={}
for file in os.listdir(path):
    file='1_Iris.xlsx'
    logger.info("**********************{}********************".format(file))
    data = pd.read_excel(os.path.join(path, file), sheet_name="dataset")
    dt = np.array(data.values)
    data = dt.astype('float')
    originData = data[:, :-1]
    target = data[:, -1]
    # file = "boston"
    # originData, target = load_boston(return_X_y=True)
    for missPattern in ['normal', 'block', 'taxa', 'chara']:
    # for missPattern in ['normal', 'block']:
        # 缺失比例只到0.5
        half = []
        methed_names_half = ['mice_misc', 'ii_misc', 'median_misc', 'random_misc', 'mida_misc', 'gain_misc']
        # 缺失比例到0.9
        all = []
        methed_names_all = ['mice_misc', 'ii_misc', 'median_misc', 'random_misc']
        mice_misc = [[] for _ in range(3)]
        ii_misc = [[] for _ in range(3)]
        median_misc = [[] for _ in range(3)]
        random_misc = [[] for _ in range(3)]
        mida_misc = [[] for _ in range(3)]
        gain_misc = [[] for _ in range(3)]
        for first_imputed_method in ['ii', 'mice']:
            for loss in ['MSELoss']:
                for method in ['Autoencoder', 'ResAutoencoder',  'StockedAutoencoder',
                               'StockedResAutoencoder']:
                    varname = "{}_{}_{}".format(first_imputed_method, loss, method)
                    globals()[varname] = [[] for _ in range(3)]
                    methed_names_half.append(varname)
                    methed_names_all.append(varname)


        for i in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # for i in [0.05, 0.1]:
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

            for first_imputed_method in ['ii','mice']:
                for loss in ['MSELoss']:
                    for method in ['Autoencoder','ResAutoencoder','StockedAutoencoder','StockedResAutoencoder']:
                        varname = "{}_{}_{}".format(first_imputed_method, loss, method)
                        try:
                            start=time.time()
                            imputedData, first_imputedData = TAI(first_imputation_method=first_imputed_method,
                                                                   batch_size=len(missData),
                                                                   epochs=300,
                                                                   theta=int(len(missData[0]) / 2),
                                                                   iterations=30,
                                                                   Autoencoder_method=method,
                                                                   loss=loss,
                                                                   use_cuda=False
                                                                   ).complete(missData)
                            logger.info("训练耗时:{}".format(time.time()-start))
                            score = RMSE(originData, imputedData)
                            score1 = RMSE(originData, first_imputedData)
                            logger.info("{}_{}_{}_{}_{} first missing rate:{},RMSE:{}".format(file,missPattern,first_imputed_method,loss,method,i, score1))
                            logger.info("{}_{}_{}_{}_{} missing rate:{},RMSE:{}".format(file,missPattern,first_imputed_method,loss,method,i, score))


                            globals()[varname][0].append(score)
                            globals()[varname][1].append(MAE(originData, imputedData))
                            globals()[varname][2].append(masked_mape_np(originData, imputedData))
                        except Exception as e:
                            logger.error(e)
                            globals()[varname][0].append(np.nan)
                            globals()[varname][1].append(np.nan)
                            globals()[varname][2].append(np.nan)

        #将三个指标在各个缺失状态下的结果求和
        logger.error("*" * 30)
        logger.error("file:{}".format(file))
        logger.error("pattern :{}".format(missPattern))
        for varname in methed_names_half:
            half.append([sum(globals()[varname][0][0:3]), sum(globals()[varname][1][0:3]),
                         sum(globals()[varname][2][0:3])])
            logger.error("half {} rmse:{} ,MAE:{},MAPE:{}".format(varname, sum(globals()[varname][0][0:3]),
                                                                  sum(globals()[varname][1][0:3]),
                                                                  sum(globals()[varname][2][0:3])))
        for varname in methed_names_all:
            all.append([sum(globals()[varname][0]), sum(globals()[varname][1]),
                        sum(globals()[varname][2])])
            logger.error("all {} rmse:{} ,MAE:{},MAPE:{}".format(varname, sum(globals()[varname][0]),
                                                                 sum(globals()[varname][1]),
                                                                 sum(globals()[varname][2])))

        #统计表现最好的方法
        half_np=np.array(half)
        all_np = np.array(all)
        #记录一个数据集，一个缺失模式下的最优情况
        once_total_result_half={}
        once_total_result_all={}
        for index in range(3):
            min_index=np.argmin(half_np[:,index])
            if methed_names_half[min_index] not in once_total_result_half.keys():
                once_total_result_half[methed_names_half[min_index]]=1
            else:
                once_total_result_half[methed_names_half[min_index]] += 1

            min_index = np.argmin(all_np[:, index])
            if methed_names_all[min_index] not in once_total_result_all.keys():
                once_total_result_all[methed_names_all[min_index]] = 1
            else:
                once_total_result_all[methed_names_all[min_index]] += 1

        #将结果打印到excel表
        value = [['Method', 'pattern', 'RMSE', 'MAE', "MAPE", 'Winner times']]
        for varname in methed_names_half:
            value.append([varname + '-half', missPattern,
                          sum(globals()[varname][0][0:3]),
                          sum(globals()[varname][1][0:3]),
                          sum(globals()[varname][2][0:3]),
                          0 if varname not in once_total_result_half else once_total_result_half[varname]])
        value.append(['--' * 4])
        for varname in methed_names_all:
            value.append([varname + '-all', missPattern,
                          sum(globals()[varname][0]),
                          sum(globals()[varname][1]),
                          sum(globals()[varname][2]),
                          0 if varname not in once_total_result_all else once_total_result_all[varname]])

        try:
            write_excel_xls_append(os.path.join(save_path,'{}.csv'.format(file)),value)
        except:
            write_excel_xls(os.path.join(save_path,'{}.csv'.format(file)),'data',value)
        logger.warning("pattern:{}".format(missPattern))
        logger.warning("best imptation half:{}".format(once_total_result_half))
        logger.warning("best imptation all:{}".format(once_total_result_all))
        logger.warning("detail:")
        x=PrettyTable(['Method','pattern','size','RMSE','MAE',"MAPE"])
        for index,names in enumerate(methed_names_half):
            try:
                temp = ["{:.4f}".format(i) for i in half[index]]
            except:
                temp = ['NaN' for _ in half[index]]
            x.add_row([names] + [missPattern]+['half'] + temp)
            logger.warning(x)
        x = PrettyTable(['Method', 'pattern', 'size', 'RMSE', 'MAE', "MAPE"])
        for index,names in enumerate(methed_names_all):
            try:
                temp = ["{:.4f}".format(i) for i in all[index]]
            except:
                temp = ['NaN' for _ in all[index]]
            x.add_row([names] + [missPattern]+['all'] + temp)
            logger.warning(x)
        total_result_half=merge_two_dicts(total_result_half,once_total_result_half)
        total_result_all=merge_two_dicts(total_result_all,once_total_result_all)
    logger.warning("total best imptation all:{}".format(total_result_all))
    logger.warning("total best imptation half:{}".format(total_result_half))
