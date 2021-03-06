"""
@author: liuAmon
@contact:utopfish@163.com
@file: print_picture.py
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
    originData = data[:, :-1]
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
        missData = gene_missingdata(rate=i, data=originData)
        try:
            imputedData = knnimput.KNN(k=3).complete(missData)
            score = evaluate.RMSE(originData, imputedData)
            knn_rmse.append(score)
            logger.info("knn missing rate:{},RMSE:{}".format(i, score))
        except:
            knn_rmse.append(np.nan)
        try:
            imputedData = mice.MICE().complete(missData)
            score = evaluate.RMSE(originData, imputedData)
            mice_rmse.append(score)
            logger.info("MICE missing rate:{},RMSE:{}".format(i, score))
        except:
            mice_rmse.append(np.nan)
        try:
            imputedData = EM().complete(missData)
            score = evaluate.RMSE(originData, imputedData)
            em_rmse.append(score)
            logger.info("EM missing rate:{},RMSE:{}".format(i, score))
        except:
            em_rmse.append(np.nan)
        try:
            imputedData = BiScaler().fit_transform(missData)
            imputedData = SoftImpute().fit_transform(imputedData)
            score = evaluate.RMSE(originData, imputedData)
            fi_bs_rmse.append(score)
            logger.info("fi BiScaler  missing rate:{},RMSE:{}".format(i, score))
        except:
            fi_bs_rmse.append(np.nan)
        try:
            imputedData = SoftImpute().fit_transform(missData)
            score = evaluate.RMSE(originData, imputedData)
            fi_si_rmse.append(score)
            logger.info("fi SoftImpute missing rate:{},RMSE:{}".format(i, score))
        except:
            fi_si_rmse.append(np.nan)
        try:
            imputedData = IterativeImputer().fit_transform(missData)
            score = evaluate.RMSE(originData, imputedData)
            fi_ii_rmse.append(score)
            logger.info("fi IterativeImputer missing rate:{},RMSE:{}".format(i, score))
        except:
            fi_ii_rmse.append(np.nan)

        try:
            imputedData = SimpleFill().fit_transform(missData)
            score = evaluate.RMSE(originData, imputedData)
            fi_sf_rmse.append(score)
            logger.info("fi SimpleFill missing rate:{},RMSE:{}".format(i, score))
        except:
            fi_sf_rmse.append(np.nan)
        try:
            imputedData = SimpleFill("median").fit_transform(missData)
            score = evaluate.RMSE(originData, imputedData)
            fi_median_rmse.append(score)
            logger.info("fi median missing rate:{},RMSE:{}".format(i, score))
        except:
            fi_median_rmse.append(np.nan)
        try:
            imputedData = impyute.imputation.cs.random(missData)
            score = evaluate.RMSE(originData, imputedData)
            random_rmse.append(score)
            logger.info("random missing rate:{},RMSE:{}".format(i, score))
        except:
            random_rmse.append(np.nan)

        try:
            imputedData = impyute.imputation.cs.mean(missData)
            score = evaluate.RMSE(originData, imputedData)
            mean_rmse.append(score)
            logger.info("mean missing rate:{},RMSE:{}".format(i, score))
        except:
            mean_rmse.append(np.nan)
        try:
            imputedData = impyute.imputation.cs.mode(missData)
            score = evaluate.RMSE(originData, imputedData)
            mode_rmse.append(score)
            logger.info("mode missing rate:{},RMSE:{}".format(i, score))
        except:
            mode_rmse.append(np.nan)
        try:
            imputedData = impyute.imputation.cs.median(missData)
            score = evaluate.RMSE(originData, imputedData)
            median_rmse.append(score)
            logger.info("median missing rate:{},RMSE:{}".format(i, score))
        except:
            median_rmse.append(np.nan)
        try:
            imputedData = MIDA().complete(missData)
            score = evaluate.RMSE(originData, imputedData)
            logger.info("MIDA missing rate:{},RMSE:{}".format(i, score))
            mida_rmse.append(score)
        except:
            mida_rmse.append(np.nan)

        try:
            imputedData = GAIN().complete(missData)
            score = evaluate.RMSE(originData, imputedData)
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

