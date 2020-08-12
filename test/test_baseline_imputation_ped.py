"""
@author: liuAmon
@contact:utopfish@163.com
@file: test_baseline_imputation_ped.py
@time: 2020/7/22 23:41
"""
import os
import time
import impyute
import numpy as np
import pandas as pd
from logger import logger
from utils.base_tools import modifier,merge_two_dicts
from utils.read_file import write_excel_xls,write_excel_xls_append

from prettytable import PrettyTable
from ycimpute.utils import evaluate
from matplotlib import pyplot as plt
from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias
from dnn.mida import MIDA
from dnn.gain import GAIN
from dnn.tai_test import TAI
from utils.misc_utils import RMSE, MAE, masked_mape_np,TF
from utils.base_impute import random_inpute
from utils.base_tools import shear_dile
from utils.read_file import readNex
from ycimpute.imputer import knnimput, mice, EM
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler, SimpleFill

path = r'../nexus_files'
pciturePath = r'../result/picture'
save_path=r'../result/ped_s'
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
    origin_data = random_inpute(data)
    for miss_pat in ['block', 'normal', 'taxa', 'chara']:
        # 缺失比例只到0.5
        half = []
        methed_names_half = ['mice_misc', 'ii_misc', 'median_misc', 'random_misc', 'mida_misc', 'gain_misc']
        # 缺失比例到0.9
        all = []
        methed_names_all = ['mice_misc', 'ii_misc', 'median_misc', 'random_misc']
        mice_misc = [[] for _ in range(4)]
        ii_misc = [[] for _ in range(4)]
        median_misc = [[] for _ in range(4)]
        random_misc = [[] for _ in range(4)]
        mida_misc = [[] for _ in range(4)]
        gain_misc = [[] for _ in range(4)]

        for first_imputed_method in ['ii', 'mice']:
            for loss in ['MSELoss']:
                for method in ['Autoencoder', 'ResAutoencoder', 'StockedAutoencoder',
                               'StockedResAutoencoder']:
                    varname = "{}_{}_{}".format(first_imputed_method, loss, method)
                    globals()[varname] = [[] for _ in range(4)]
                    methed_names_half.append(varname)
                    methed_names_all.append(varname)


        # for i in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for i in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            if miss_pat == 'normal':
                miss_data = gene_missingdata(rate=i, data=origin_data)
            elif miss_pat == 'taxa':
                miss_data = gene_missingdata_taxa_bias(rate=i, data=origin_data)
            elif miss_pat == 'chara':
                miss_data = gene_missingdata_chara_bias(rate=i, data=origin_data)
            elif miss_pat == 'block':
                miss_data = gene_missingdata_block_bias(rate=i, data=origin_data)
            else:
                raise Exception("缺失模式错误，请在'normal','taxa','chara','block'中选择对应模式")

            mark = [temp[0] for temp in pd.DataFrame(np.unique(miss_data)).dropna(axis=0).values]

            try:
                imputed_data = mice.MICE().complete(miss_data)
                imputed_data = modifier(imputed_data, mark)
                score = evaluate.RMSE(origin_data, imputed_data)
                mice_misc[0].append(score)
                mice_misc[1].append(MAE(origin_data, imputed_data))
                mice_misc[2].append(masked_mape_np(origin_data, imputed_data))
                mice_misc[3].append(TF(origin_data, imputed_data))
                logger.info("MICE missing rate:{},RMSE:{}".format(i, score))
            except:
                mice_misc[0].append(np.inf)
                mice_misc[1].append(np.inf)
                mice_misc[2].append(np.inf)
                mice_misc[3].append(np.inf)
            try:
                imputed_data = IterativeImputer().fit_transform(miss_data)
                imputed_data = modifier(imputed_data, mark)
                score = evaluate.RMSE(origin_data, imputed_data)
                ii_misc[0].append(score)
                ii_misc[1].append(MAE(origin_data, imputed_data))
                ii_misc[2].append(masked_mape_np(origin_data, imputed_data))
                ii_misc[3].append(TF(origin_data, imputed_data))
                logger.info("fi IterativeImputer missing rate:{},RMSE:{}".format(i, score))
            except:
                ii_misc[0].append(np.inf)
                ii_misc[1].append(np.inf)
                ii_misc[2].append(np.inf)
                ii_misc[3].append(np.inf)
            try:
                imputed_data = SimpleFill("median").fit_transform(miss_data)
                imputed_data = modifier(imputed_data, mark)
                score = evaluate.RMSE(origin_data, imputed_data)
                median_misc[0].append(score)
                median_misc[1].append(MAE(origin_data, imputed_data))
                median_misc[2].append(masked_mape_np(origin_data, imputed_data))
                median_misc[3].append(TF(origin_data, imputed_data))
                logger.info("fi median missing rate:{},RMSE:{}".format(i, score))
            except:
                median_misc[0].append(np.inf)
                median_misc[1].append(np.inf)
                median_misc[2].append(np.inf)
                median_misc[3].append(np.inf)
            try:
                imputed_data = impyute.imputation.cs.random(miss_data)
                imputed_data = modifier(imputed_data, mark)
                score = evaluate.RMSE(origin_data, imputed_data)
                random_misc[0].append(score)
                random_misc[1].append(MAE(origin_data, imputed_data))
                random_misc[2].append(masked_mape_np(origin_data, imputed_data))
                random_misc[3].append(TF(origin_data, imputed_data))
                logger.info("random missing rate:{},RMSE:{}".format(i, score))
            except:
                random_misc[0].append(np.inf)
                random_misc[1].append(np.inf)
                random_misc[2].append(np.inf)
                random_misc[3].append(np.inf)
            try:
                imputed_data = MIDA().complete(miss_data)
                imputed_data = modifier(imputed_data, mark)
                score = evaluate.RMSE(origin_data, imputed_data)
                logger.info("MIDA missing rate:{},RMSE:{}".format(i, score))
                mida_misc[0].append(score)
                mida_misc[1].append(MAE(origin_data, imputed_data))
                mida_misc[2].append(masked_mape_np(origin_data, imputed_data))
                mida_misc[3].append(TF(origin_data, imputed_data))
            except:
                mida_misc[0].append(np.inf)
                mida_misc[1].append(np.inf)
                mida_misc[2].append(np.inf)
                mida_misc[3].append(np.inf)
            try:
                imputed_data = GAIN().complete(miss_data)
                imputed_data = modifier(imputed_data, mark)
                score = evaluate.RMSE(origin_data, imputed_data)
                logger.info("GAIN missing rate:{},RMSE:{}".format(i, score))
                gain_misc[0].append(score)
                gain_misc[1].append(MAE(origin_data, imputed_data))
                gain_misc[2].append(masked_mape_np(origin_data, imputed_data))
                gain_misc[3].append(TF(origin_data, imputed_data))
            except:
                gain_misc[0].append(np.inf)
                gain_misc[1].append(np.inf)
                gain_misc[2].append(np.inf)
                gain_misc[3].append(np.inf)
            for first_imputed_method in ['ii', 'mice']:
                    for loss in ['MSELoss']:
                        for method in ['Autoencoder', 'ResAutoencoder',
                                       'StockedAutoencoder','StockedResAutoencoder']:
                            varname = "{}_{}_{}".format(first_imputed_method, loss, method)
                            try:
                                start = time.time()
                                imputed_data, first_imputed_data = TAI(first_imputation_method=first_imputed_method,
                                                                       batch_size=len(miss_data),
                                                                       epochs=300,
                                                                       theta=int(len(miss_data[0]) / 2),
                                                                       iterations=30,
                                                                       Autoencoder_method=method,
                                                                       loss=loss,
                                                                       use_cuda=False
                                                                       ).complete(miss_data)
                                logger.info("训练耗时:{}".format(time.time() - start))
                                imputed_data = modifier(imputed_data, mark)
                                score = RMSE(origin_data, imputed_data)
                                score1 = RMSE(origin_data, first_imputed_data)
                                logger.info("{}_{}_{}_{}_{} first missing rate:{},RMSE:{}".format(file, miss_pat,
                                                                                                  first_imputed_method,
                                                                                                  loss, method, i,
                                                                                                  score1))
                                logger.info("{}_{}_{}_{}_{} missing rate:{},RMSE:{}".format(file, miss_pat,
                                                                                            first_imputed_method, loss,
                                                                                            method, i, score))

                                globals()[varname][0].append(score)
                                globals()[varname][1].append(MAE(origin_data, imputed_data))
                                globals()[varname][2].append(masked_mape_np(origin_data, imputed_data))
                                globals()[varname][3].append(TF(origin_data, imputed_data))
                            except Exception as e:
                                logger.error(e)
                                globals()[varname][0].append(np.inf)
                                globals()[varname][1].append(np.inf)
                                globals()[varname][2].append(np.inf)
                                globals()[varname][3].append(np.inf)

        # 将三个指标在各个缺失状态下的结果求和
        logger.error("*" * 30)
        logger.error("file:{}".format(file))
        logger.error("pattern :{}".format(miss_pat))
        for varname in methed_names_half:
            half.append([sum(globals()[varname][0][0:3]), sum(globals()[varname][1][0:3]),
                         sum(globals()[varname][2][0:3]),sum(globals()[varname][3][0:3])])
            logger.error("half {} rmse:{} ,MAE:{},MAPE:{}".format(varname, sum(globals()[varname][0][0:3]),
                                                                  sum(globals()[varname][1][0:3]),
                                                                  sum(globals()[varname][2][0:3]),
                                                                  sum(globals()[varname][3][0:3])))
        for varname in methed_names_all:
            all.append([sum(globals()[varname][0]), sum(globals()[varname][1]),
                        sum(globals()[varname][2]),sum(globals()[varname][3])])
            logger.error("all {} rmse:{} ,MAE:{},MAPE:{}".format(varname, sum(globals()[varname][0]),
                                                                 sum(globals()[varname][1]),
                                                                 sum(globals()[varname][2]),
                                                                 sum(globals()[varname][3])))

        # 统计表现最好的方法
        half_np = np.array(half)
        all_np = np.array(all)
        # 记录一个数据集，一个缺失模式下的最优情况
        once_total_result_half = {}
        once_total_result_all = {}
        for index in range(4):
            min_index = np.argmin(half_np[:, index])
            if methed_names_half[min_index] not in once_total_result_half.keys():
                once_total_result_half[methed_names_half[min_index]] = 1
            else:
                once_total_result_half[methed_names_half[min_index]] += 1

            min_index = np.argmin(all_np[:, index])
            if methed_names_all[min_index] not in once_total_result_all.keys():
                once_total_result_all[methed_names_all[min_index]] = 1
            else:
                once_total_result_all[methed_names_all[min_index]] += 1

        # 将结果打印到excel表
        value = [['Method', 'pattern', 'RMSE', 'MAE', "MAPE", "TF",'Winner times']]
        for varname in methed_names_half:
            value.append([varname + '-half', miss_pat,
                          sum(globals()[varname][0][0:3]),
                          sum(globals()[varname][1][0:3]),
                          sum(globals()[varname][2][0:3]),
                          sum(globals()[varname][3][0:3]),
                          0 if varname not in once_total_result_half else once_total_result_half[varname]])
        value.append(['--' * 4])
        for varname in methed_names_all:
            value.append([varname + '-all', miss_pat,
                          sum(globals()[varname][0]),
                          sum(globals()[varname][1]),
                          sum(globals()[varname][2]),
                          sum(globals()[varname][3]),
                          0 if varname not in once_total_result_all else once_total_result_all[varname]])

        try:
            write_excel_xls_append(os.path.join(save_path, '{}.csv'.format(file)), value)
        except:
            write_excel_xls(os.path.join(save_path, '{}.csv'.format(file)), 'data', value)
        logger.warning("pattern:{}".format(miss_pat))
        logger.warning("best imptation half:{}".format(once_total_result_half))
        logger.warning("best imptation all:{}".format(once_total_result_all))
        logger.warning("detail:")
        x = PrettyTable(['Method', 'pattern', 'size', 'RMSE', 'MAE', "MAPE","TF"])
        for index, names in enumerate(methed_names_half):
            try:
                temp = ["{:.4f}".format(i) for i in half[index]]
            except:
                temp = ['NaN' for _ in half[index]]
            x.add_row([names] + [miss_pat] + ['half'] + temp)
            logger.warning(x)
        x = PrettyTable(['Method', 'pattern', 'size', 'RMSE', 'MAE', "MAPE","TF"])
        for index, names in enumerate(methed_names_all):
            try:
                temp = ["{:.4f}".format(i) for i in all[index]]
            except:
                temp = ['NaN' for _ in all[index]]
            x.add_row([names] + [miss_pat] + ['all'] + temp)
            logger.warning(x)
        total_result_half = merge_two_dicts(total_result_half, once_total_result_half)
        total_result_all = merge_two_dicts(total_result_all, once_total_result_all)
        logger.warning("total best imptation all:{}".format(total_result_all))
        logger.warning("total best imptation half:{}".format(total_result_half))
