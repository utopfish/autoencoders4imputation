"""
@author: liuAmon
@contact:utopfish@163.com
@file: test_my_method.py
@time: 2020/7/17 1:01
"""
import os
import impyute
import numpy as np
import pandas as pd

from logger import logger
from ycimpute.utils import evaluate
from utils.misc_utils import RMSE, MAE, masked_mape_np
from matplotlib import pyplot as plt
from utils.handle_missingdata import gene_missingdata, gene_missingdata_taxa_bias, gene_missingdata_chara_bias, \
    gene_missingdata_block_bias
from dnn.mida import MIDA
from dnn.gain import GAIN
from dnn.tai import TAI, TResAI
from ycimpute.imputer import knnimput, mice, EM
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler, SimpleFill
from sklearn.datasets import load_boston

path = r'../public_data/'
pciturePath = r'E:\labCode\picture_save\3_misc'
for file in os.listdir(path):
    # logger.info("**********************{}********************".format(file))
    # data = pd.read_excel(os.path.join(path, file), sheet_name="dataset")
    # dt = np.array(data.values)
    # data = dt.astype('float')
    # origin_data = data[:, :-1]
    # target = data[:, -1]
    file = "boston"
    origin_data, target = load_boston(return_X_y=True)
    # for miss_pat in ['normal','taxa','chara','block']:
    for miss_pat in ['normal', 'block', 'taxa', 'chara']:
        mice_misc = [[] for _ in range(3)]
        ii_misc = [[] for _ in range(3)]
        median_misc = [[] for _ in range(3)]
        random_misc = [[] for _ in range(3)]
        mida_misc = [[] for _ in range(3)]
        gain_misc = [[] for _ in range(3)]
        tai_ii_misc = [[] for _ in range(3)]
        tai_mice_misc = [[] for _ in range(3)]
        tai_random_misc = [[] for _ in range(3)]
        tai_none_misc = [[] for _ in range(3)]
        tresai_mice_misc = [[] for _ in range(3)]
        tresai_ii_misc = [[] for _ in range(3)]
        for i in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
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
            try:
                imputed_data = mice.MICE().complete(miss_data)
                score = evaluate.RMSE(origin_data, imputed_data)
                mice_misc[0].append(score)
                mice_misc[1].append(MAE(origin_data, imputed_data))
                mice_misc[2].append(masked_mape_np(origin_data, imputed_data))
                logger.info("MICE missing rate:{},RMSE:{}".format(i, score))
            except:
                mice_misc[0].append(np.nan)
                mice_misc[1].append(np.nan)
                mice_misc[2].append(np.nan)
            try:
                imputed_data = IterativeImputer().fit_transform(miss_data)
                score = evaluate.RMSE(origin_data, imputed_data)
                ii_misc[0].append(score)
                ii_misc[1].append(MAE(origin_data, imputed_data))
                ii_misc[2].append(masked_mape_np(origin_data, imputed_data))
                logger.info("fi IterativeImputer missing rate:{},RMSE:{}".format(i, score))
            except:
                ii_misc[0].append(np.nan)
                ii_misc[1].append(np.nan)
                ii_misc[2].append(np.nan)
            try:
                imputed_data = SimpleFill("median").fit_transform(miss_data)
                score = evaluate.RMSE(origin_data, imputed_data)
                median_misc[0].append(score)
                median_misc[1].append(MAE(origin_data, imputed_data))
                median_misc[2].append(masked_mape_np(origin_data, imputed_data))
                logger.info("fi median missing rate:{},RMSE:{}".format(i, score))
            except:
                median_misc[0].append(np.nan)
                median_misc[1].append(np.nan)
                median_misc[2].append(np.nan)
            try:
                imputed_data = impyute.imputation.cs.random(miss_data)
                score = evaluate.RMSE(origin_data, imputed_data)
                random_misc[0].append(score)
                random_misc[1].append(MAE(origin_data, imputed_data))
                random_misc[2].append(masked_mape_np(origin_data, imputed_data))
                logger.info("random missing rate:{},RMSE:{}".format(i, score))
            except:
                random_misc[0].append(np.nan)
                random_misc[1].append(np.nan)
                random_misc[2].append(np.nan)
            try:
                imputed_data = MIDA().complete(miss_data)
                score = evaluate.RMSE(origin_data, imputed_data)
                logger.info("MIDA missing rate:{},RMSE:{}".format(i, score))
                mida_misc[0].append(score)
                mida_misc[1].append(MAE(origin_data, imputed_data))
                mida_misc[2].append(masked_mape_np(origin_data, imputed_data))
            except:
                mida_misc[0].append(np.nan)
                mida_misc[1].append(np.nan)
                mida_misc[2].append(np.nan)
            try:
                imputed_data = GAIN().complete(miss_data)
                score = evaluate.RMSE(origin_data, imputed_data)
                logger.info("GAIN missing rate:{},RMSE:{}".format(i, score))
                gain_misc[0].append(score)
                gain_misc[1].append(MAE(origin_data, imputed_data))
                gain_misc[2].append(masked_mape_np(origin_data, imputed_data))
            except:
                gain_misc[0].append(np.nan)
                gain_misc[1].append(np.nan)
                gain_misc[2].append(np.nan)
            try:
                imputed_data, first_imputed_data = TAI().complete(miss_data)
                score = evaluate.RMSE(origin_data, imputed_data)
                score1 = evaluate.RMSE(origin_data, first_imputed_data)
                logger.info("TAI ii first missing rate:{},RMSE:{}".format(i, score1))
                logger.info("TAI ii missing rate:{},RMSE:{}".format(i, score))
                tai_ii_misc[0].append(score)
                tai_ii_misc[1].append(MAE(origin_data, imputed_data))
                tai_ii_misc[2].append(masked_mape_np(origin_data, imputed_data))
            except:
                tai_ii_misc[0].append(np.nan)
                tai_ii_misc[1].append(np.nan)
                tai_ii_misc[2].append(np.nan)

            try:
                imputed_data, first_imputed_data = TAI(first_imputation_method='mice', batch_size=len(miss_data),
                                                       epochs=300, theta=int(len(miss_data[0]) / 2),
                                                       iterations=30).complete(miss_data)
                score = evaluate.RMSE(origin_data, imputed_data)
                score1 = evaluate.RMSE(origin_data, first_imputed_data)
                logger.info("TAI mice first missing rate:{},RMSE:{}".format(i, score1))
                logger.info("TAI mice missing rate:{},RMSE:{}".format(i, score))

                tai_mice_misc[0].append(score)
                tai_mice_misc[1].append(MAE(origin_data, imputed_data))
                tai_mice_misc[2].append(masked_mape_np(origin_data, imputed_data))
            except Exception as e:
                logger.error(e)
                tai_mice_misc[0].append(np.nan)
                tai_mice_misc[1].append(np.nan)
                tai_mice_misc[2].append(np.nan)

            try:
                imputed_data, first_imputed_data = TResAI(first_imputation_method='mice', batch_size=len(miss_data),
                                                          epochs=300, theta=int(len(miss_data[0]) / 2),
                                                          iterations=30).complete(miss_data)
                score = evaluate.RMSE(origin_data, imputed_data)
                score1 = evaluate.RMSE(origin_data, first_imputed_data)
                logger.info("TResAI mice first missing rate:{},RMSE:{}".format(i, score1))
                logger.info("TResAI mice missing rate:{},RMSE:{}".format(i, score))
                tresai_mice_misc[0].append(score)
                tresai_mice_misc[1].append(MAE(origin_data, imputed_data))
                tresai_mice_misc[2].append(masked_mape_np(origin_data, imputed_data))
            except Exception as e:
                logger.error(e)
                tresai_mice_misc[0].append(np.nan)
                tresai_mice_misc[1].append(np.nan)
                tresai_mice_misc[2].append(np.nan)

            try:
                imputed_data, first_imputed_data = TResAI(batch_size=len(miss_data),
                                                          epochs=300, theta=int(len(miss_data[0]) / 2),
                                                          iterations=30).complete(miss_data)
                score = evaluate.RMSE(origin_data, imputed_data)
                score1 = evaluate.RMSE(origin_data, first_imputed_data)
                logger.info("TResAI ii first missing rate:{},RMSE:{}".format(i, score1))
                logger.info("TResAI ii missing rate:{},RMSE:{}".format(i, score))
                tresai_ii_misc[0].append(score)
                tresai_ii_misc[1].append(MAE(origin_data, imputed_data))
                tresai_ii_misc[2].append(masked_mape_np(origin_data, imputed_data))
            except Exception as e:
                logger.error(e)
                tresai_ii_misc[0].append(np.nan)
                tresai_ii_misc[1].append(np.nan)
                tresai_ii_misc[2].append(np.nan)
            # try:
            #     imputed_data , first_imputed_data= TAI(first_imputation_method='random',batch_size=len(miss_data),epochs=300,theta=int(len(miss_data[0])/2),iterations=30).complete(miss_data)
            #     score = evaluate.RMSE(origin_data, imputed_data)
            #     score1 = evaluate.RMSE(origin_data, first_imputed_data)
            #     logger.info("TAI random first missing rate:{},RMSE:{}".format(i, score1))
            #     logger.info("TAI random missing rate:{},RMSE:{}".format(i, score))
            #     tai_random_misc.append(score)
            # except Exception as e:
            #     logger.error(e)
            #     tai_random_misc.append(np.nan)
            #
            # try:
            #     imputed_data , first_imputed_data= TAI(first_imputation_method='None',batch_size=len(miss_data),epochs=500,theta=int(len(miss_data[0])/2),iterations=100).complete(miss_data)
            #     score = evaluate.RMSE(origin_data, imputed_data)
            #     score1 = evaluate.RMSE(origin_data, first_imputed_data)
            #     logger.info("TAI none first missing rate:{},RMSE:{}".format(i, score1))
            #     logger.info("TAI none missing rate:{},RMSE:{}".format(i, score))
            #     tai_none_misc.append(score)
            # except Exception as e:
            #     logger.error(e)
            #     tai_none_misc.append(np.nan)

        color = ['blue', 'green', 'red', 'yellow', 'black', 'burlywood', 'cadetblue', 'chartreuse', 'purple', 'coral',
                 'aqua', 'aquamarine', 'darkblue', 'y', 'm']
        plt.figure()
        x = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        plt.plot(x, mice_misc[0], color=color[1], label='mice')
        plt.plot(x, ii_misc[0], color=color[2], label='ii')
        plt.plot(x, median_misc[0], color=color[3], linestyle=':', label='median')
        plt.plot(x, random_misc[0], color=color[9], linestyle=':', label='random')
        plt.plot(x, mida_misc[0], color=color[13], linewidth=3.0, linestyle='-.', label='mida')
        plt.plot(x, gain_misc[0], color=color[14], linewidth=3.0, linestyle='-.', label='gain')
        # plt.plot(x, tai_ii_misc, color=color[10], linewidth=2.0, linestyle='-', label='tai ii')
        plt.plot(x, tai_mice_misc[0], color=color[12], linewidth=2.0, linestyle='-', label='tai mice')
        # plt.plot(x, tai_random_misc, color=color[0], linewidth=2.0, linestyle='-', label='tai random')
        plt.plot(x, tresai_mice_misc[0], color=color[3], linewidth=2.0, linestyle='-', label='tresai mice')
        plt.title("rmse of different missing rate in {}_{}".format(file, miss_pat))
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(pciturePath, "rmse_of_different_missing_rate_in_{}_{}.png".format(file, miss_pat)))
        plt.show()
        print("*" * 20)

        plt.figure()
        x = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        plt.plot(x, mice_misc[1], color=color[1], label='mice')
        plt.plot(x, ii_misc[1], color=color[2], label='ii')
        plt.plot(x, median_misc[1], color=color[3], linestyle=':', label='median')
        plt.plot(x, random_misc[1], color=color[9], linestyle=':', label='random')
        plt.plot(x, mida_misc[1], color=color[13], linewidth=3.0, linestyle='-.', label='mida')
        plt.plot(x, gain_misc[1], color=color[14], linewidth=3.0, linestyle='-.', label='gain')
        # plt.plot(x, tai_ii_misc, color=color[10], linewidth=2.0, linestyle='-', label='tai ii')
        plt.plot(x, tai_mice_misc[1], color=color[12], linewidth=2.0, linestyle='-', label='tai mice')
        # plt.plot(x, tai_random_misc, color=color[0], linewidth=2.0, linestyle='-', label='tai random')
        plt.plot(x, tresai_mice_misc[1], color=color[3], linewidth=2.0, linestyle='-', label='tresai mice')
        plt.title("MAE of different missing rate in {}_{}".format(file, miss_pat))
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(pciturePath, "MAE_of_different_missing_rate_in_{}_{}.png".format(file, miss_pat)))
        plt.show()
        print("*" * 20)

        plt.figure()
        x = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        plt.plot(x, mice_misc[2], color=color[1], label='mice')
        plt.plot(x, ii_misc[2], color=color[2], label='ii')
        plt.plot(x, median_misc[2], color=color[3], linestyle=':', label='median')
        plt.plot(x, random_misc[2], color=color[9], linestyle=':', label='random')
        plt.plot(x, mida_misc[2], color=color[13], linewidth=3.0, linestyle='-.', label='mida')
        plt.plot(x, gain_misc[2], color=color[14], linewidth=3.0, linestyle='-.', label='gain')
        # plt.plot(x, tai_ii_misc, color=color[10], linewidth=2.0, linestyle='-', label='tai ii')
        plt.plot(x, tai_mice_misc[2], color=color[12], linewidth=2.0, linestyle='-', label='tai mice')
        # plt.plot(x, tai_random_misc, color=color[0], linewidth=2.0, linestyle='-', label='tai random')
        plt.plot(x, tresai_mice_misc[2], color=color[3], linewidth=2.0, linestyle='-', label='tresai mice')
        plt.title("mape of different missing rate in {}_{}".format(file, miss_pat))
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(pciturePath, "mape_of_different_missing_rate_in_{}_{}.png".format(file, miss_pat)))
        plt.show()
        print("*" * 20)

        logger.error("*" * 30)
        logger.error("file:{}".format(file))
        logger.error("pattern :{}".format(miss_pat))
        logger.error("0.05-0.3")
        logger.error("mice rmse:{} ,MAE:{},MAPE:{}".format(sum(mice_misc[0][0:3]), sum(mice_misc[1][0:3]),
                                                           sum(mice_misc[2][0:3])))
        logger.error(
            "ii_rmse:{},MAE:{},MAPE:{}".format(sum(ii_misc[0][0:3]), sum(ii_misc[1][0:3]), sum(ii_misc[2][0:3])))
        logger.error("median_rmse:{},MAE:{},MAPE:{}".format(sum(median_misc[0][0:3]), sum(median_misc[1][0:3]),
                                                            sum(median_misc[2][0:3])))
        logger.error("random_rmse:{},MAE:{},MAPE:{}".format(sum(random_misc[0][0:3]), sum(random_misc[1][0:3]),
                                                            sum(random_misc[2][0:3])))
        logger.error("mida_rmse:{},MAE:{},MAPE:{}".format(sum(mida_misc[0][0:3]), sum(mida_misc[1][0:3]),
                                                          sum(mida_misc[2][0:3])))
        logger.error("gain_rmse:{},MAE:{},MAPE:{}".format(sum(gain_misc[0][0:3]), sum(gain_misc[1][0:3]),
                                                          sum(gain_misc[2][0:3])))
        logger.error("tai_mice_rmse:{},MAE:{},MAPE:{}".format(sum(tai_mice_misc[0][0:3]), sum(tai_mice_misc[1][0:3]),
                                                              sum(tai_mice_misc[2][0:3])))
        logger.error(
            "tresai_mice_rmse:{},MAE:{},MAPE:{}".format(sum(tresai_mice_misc[0][0:3]), sum(tresai_mice_misc[1][0:3]),
                                                        sum(tresai_mice_misc[2][0:3])))
        logger.error("tresai_ii_rmse:{},MAE:{},MAPE:{}".format(sum(tresai_ii_misc[0][0:3]), sum(tresai_ii_misc[1][0:3]),
                                                               sum(tresai_ii_misc[2][0:3])))
        logger.error("*" * 30)
        logger.error("ALL")

        logger.error("mice rmse:{} ,MAE:{},MAPE:{}".format(sum(mice_misc[0]), sum(mice_misc[1]),
                                                           sum(mice_misc[2])))
        logger.error(
            "ii_rmse:{},MAE:{},MAPE:{}".format(sum(ii_misc[0]), sum(ii_misc[1]), sum(ii_misc[2])))
        logger.error("median_rmse:{},MAE:{},MAPE:{}".format(sum(median_misc[0]), sum(median_misc[1]),
                                                            sum(median_misc[2])))
        logger.error("random_rmse:{},MAE:{},MAPE:{}".format(sum(random_misc[0]), sum(random_misc[1]),
                                                            sum(random_misc[2])))
        logger.error("tai_mice_rmse:{},MAE:{},MAPE:{}".format(sum(tai_mice_misc[0]), sum(tai_mice_misc[1]),
                                                              sum(tai_mice_misc[2])))
        logger.error(
            "tresai_mice_rmse:{},MAE:{},MAPE:{}".format(sum(tresai_mice_misc[0]), sum(tresai_mice_misc[1]),
                                                        sum(tresai_mice_misc[2])))
        logger.error("tresai_ii_rmse:{},MAE:{},MAPE:{}".format(sum(tresai_ii_misc[0]), sum(tresai_ii_misc[1]),
                                                               sum(tresai_ii_misc[2])))
