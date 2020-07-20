"""
@author: liuAmon
@contact:utopfish@163.com
@file: test_baseline_imputation_uci.py
@time: 2020/7/16 16:53
"""
import os
import impyute
import numpy as np
import pandas as pd

from logger import logger
from ycimpute.utils import evaluate
from matplotlib import pyplot as plt
from utils.handle_missingdata import gene_missingdata
from dnn.mida import MIDA
from dnn.gain import GAIN
from ycimpute.imputer import knnimput, mice, EM
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler, SimpleFill

path = r'../public_data/'
pciturePath = r'G:\labWork\imputation_plt\pub'
for file in os.listdir(path):
    logger.info("**********************{}********************".format(file))
    data = pd.read_excel(os.path.join(path, file), sheet_name="dataset")
    dt = np.array(data.values)
    data = dt.astype('float')
    origin_data = data[:, :-1]
    target = data[:, -1]
    knn_rmse = []
    mice_rmse = []
    em_rmse = []
    fi_knn_rmse = []
    fi_bs_rmse = []
    fi_si_rmse = []
    fi_ii_rmse = []
    fi_sf_rmse = []
    fi_median_rmse = []
    random_rmse = []
    mean_rmse = []
    mode_rmse = []
    median_rmse = []
    mida_rmse = []
    gain_rmse = []
    for i in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        miss_data = gene_missingdata(rate=i, data=origin_data)
        try:
            imputed_data = knnimput.KNN(k=3).complete(miss_data)
            score = evaluate.RMSE(origin_data, imputed_data)
            knn_rmse.append(score)
            logger.info("knn missing rate:{},RMSE:{}".format(i, score))
        except:
            knn_rmse.append(np.nan)
        try:
            imputed_data = mice.MICE().complete(miss_data)
            score = evaluate.RMSE(origin_data, imputed_data)
            mice_rmse.append(score)
            logger.info("MICE missing rate:{},RMSE:{}".format(i, score))
        except:
            mice_rmse.append(np.nan)
        try:
            imputed_data = EM().complete(miss_data)
            score = evaluate.RMSE(origin_data, imputed_data)
            em_rmse.append(score)
            logger.info("EM missing rate:{},RMSE:{}".format(i, score))
        except:
            em_rmse.append(np.nan)
        try:
            imputed_data = BiScaler().fit_transform(miss_data)
            imputed_data = SoftImpute().fit_transform(imputed_data)
            score = evaluate.RMSE(origin_data, imputed_data)
            fi_bs_rmse.append(score)
            logger.info("fi BiScaler  missing rate:{},RMSE:{}".format(i, score))
        except:
            fi_bs_rmse.append(np.nan)
        try:
            imputed_data = SoftImpute().fit_transform(miss_data)
            score = evaluate.RMSE(origin_data, imputed_data)
            fi_si_rmse.append(score)
            logger.info("fi SoftImpute missing rate:{},RMSE:{}".format(i, score))
        except:
            fi_si_rmse.append(np.nan)
        try:
            imputed_data = IterativeImputer().fit_transform(miss_data)
            score = evaluate.RMSE(origin_data, imputed_data)
            fi_ii_rmse.append(score)
            logger.info("fi IterativeImputer missing rate:{},RMSE:{}".format(i, score))
        except:
            fi_ii_rmse.append(np.nan)

        try:
            imputed_data = SimpleFill().fit_transform(miss_data)
            score = evaluate.RMSE(origin_data, imputed_data)
            fi_sf_rmse.append(score)
            logger.info("fi SimpleFill missing rate:{},RMSE:{}".format(i, score))
        except:
            fi_sf_rmse.append(np.nan)
        try:
            imputed_data = SimpleFill("median").fit_transform(miss_data)
            score = evaluate.RMSE(origin_data, imputed_data)
            fi_median_rmse.append(score)
            logger.info("fi median missing rate:{},RMSE:{}".format(i, score))
        except:
            fi_median_rmse.append(np.nan)
        try:
            imputed_data = impyute.imputation.cs.random(miss_data)
            score = evaluate.RMSE(origin_data, imputed_data)
            random_rmse.append(score)
            logger.info("random missing rate:{},RMSE:{}".format(i, score))
        except:
            random_rmse.append(np.nan)

        try:
            imputed_data = impyute.imputation.cs.mean(miss_data)
            score = evaluate.RMSE(origin_data, imputed_data)
            mean_rmse.append(score)
            logger.info("mean missing rate:{},RMSE:{}".format(i, score))
        except:
            mean_rmse.append(np.nan)
        try:
            imputed_data = impyute.imputation.cs.mode(miss_data)
            score = evaluate.RMSE(origin_data, imputed_data)
            mode_rmse.append(score)
            logger.info("mode missing rate:{},RMSE:{}".format(i, score))
        except:
            mode_rmse.append(np.nan)
        try:
            imputed_data = impyute.imputation.cs.median(miss_data)
            score = evaluate.RMSE(origin_data, imputed_data)
            median_rmse.append(score)
            logger.info("median missing rate:{},RMSE:{}".format(i, score))
        except:
            median_rmse.append(np.nan)
        try:
            imputed_data = MIDA().complete(miss_data)
            score = evaluate.RMSE(origin_data, imputed_data)
            logger.info("MIDA missing rate:{},RMSE:{}".format(i, score))
            mida_rmse.append(score)
        except:
            mida_rmse.append(np.nan)

        try:
            imputed_data = GAIN().complete(miss_data)
            score = evaluate.RMSE(origin_data, imputed_data)
            logger.info("MIDA missing rate:{},RMSE:{}".format(i, score))
            gain_rmse.append(score)
        except:
            gain_rmse.append(np.nan)

    color = ['blue', 'green', 'red', 'yellow', 'black', 'burlywood', 'cadetblue', 'chartreuse', 'purple', 'coral',
             'aqua', 'aquamarine', 'darkblue', 'y','m']

    plt.figure()
    x = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    plt.plot(x, knn_rmse, color=color[0], label='knn')
    plt.plot(x, mice_rmse, color=color[1], label='mice')
    plt.plot(x, em_rmse, color=color[2], label='em')
    plt.plot(x, fi_bs_rmse, color=color[4], label='BiScaler')
    plt.plot(x, fi_si_rmse, color=color[5], label='SoftImpute')
    plt.plot(x, fi_ii_rmse, color=color[6], label='IterativeImputer')
    plt.plot(x, fi_sf_rmse, color=color[7], label='mean')
    plt.plot(x, fi_median_rmse, color=color[8], label='median')
    plt.plot(x, random_rmse, color=color[9], label='random')
    plt.plot(x, mode_rmse, color=color[11], label='mode')
    plt.plot(x, mida_rmse, color=color[13], linewidth=3.0,linestyle='-.',label='mida')
    plt.plot(x, gain_rmse, color=color[14], linewidth=3.0, linestyle='-.', label='gain')
    plt.title("rmse of different missing rate in {}".format(file))
    plt.legend(loc="upper left")
    # plt.show()
    plt.savefig(os.path.join(pciturePath, "rmse_of_different_missing_rate_in_{}.png".format(file)))

