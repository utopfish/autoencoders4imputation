#@Time:2019/12/27 20:19
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:examperiment_pub.py
__author__ = "liuAmon"
import os
import numpy as np
import pandas as pd
from logger import logger
from utils.handlerLoss import MSE
from sklearn import preprocessing
from utils.handlerMissData import geneMissData
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler,SimpleFill
from iris_buldModel import get_miss_location,interpolation
if __name__=="__main__":
    for file in os.listdir('public_data'):
        for miss_rate in range(1, 10):
            miss_rate = miss_rate / 10
            try:
                if file.endswith('xlsx'):
                    iris = pd.read_excel('public_data/{}'.format(file), sheet_name="dataset")
                    dt = np.array(iris.values)
                    iris = dt.astype('float')
                    # #归一化，去掉标签
                    min_max_scaler = preprocessing.MinMaxScaler()
                    iris = min_max_scaler.fit_transform(iris[:, :-1])
                    x_test_alldata, miss_mask = geneMissData(miss_rate, iris)
                    miss_location = get_miss_location(x_test_alldata[miss_mask])
                    modelName = file + '_{}'.format(miss_rate)
                    inp = interpolation(modelName=modelName, completeData=np.delete(iris, miss_mask, axis=0))
                    if not os.path.exists('Model/{}.pkl'.format(modelName)):
                        inp.fit()
                    pred_data, fix_data = inp.prd(x_test_alldata[miss_mask])
                    beforeRevise = MSE(pred_data, iris[miss_mask])
                    afterRevise = MSE(fix_data, iris[miss_mask])

                    logger.info("{}--before revise:{}".format(modelName, beforeRevise))
                    logger.info("{}--after revise:{}".format(modelName, afterRevise))
                    logger.info("{}--propose MSE increase:{}".format(modelName, beforeRevise - afterRevise))
                    X_filled_knn = KNN(k=3).fit_transform(x_test_alldata)
                    X_filled_knn_revised = inp.revise(X_filled_knn[miss_mask], miss_location)
                    logger.info("{}--knn:{}".format(modelName, MSE(X_filled_knn[miss_mask], iris[miss_mask])))
                    logger.info("{}--knn--revised:{}".format(modelName, MSE(X_filled_knn_revised, iris[miss_mask])))
                    logger.info("{}--knn--improved:{}".format(modelName,
                                                                       MSE(X_filled_knn[miss_mask], iris[miss_mask]) - MSE(
                                                                           X_filled_knn_revised, iris[miss_mask])))

                    X_filled_ii = IterativeImputer().fit_transform(x_test_alldata)
                    X_filled_ii_revised = inp.revise(X_filled_ii[miss_mask], miss_location)
                    logger.info("{}--iterativeImputer:{}".format(modelName, MSE(X_filled_ii[miss_mask], iris[miss_mask])))
                    logger.info(
                        "{}--iterativeImputer--revised:{}".format(modelName, MSE(X_filled_ii_revised, iris[miss_mask])))
                    logger.info("{}--iterativeImputer--improve:{}".format(modelName, MSE(X_filled_ii[miss_mask],
                                                                                                  iris[miss_mask]) - MSE(
                        X_filled_ii_revised, iris[miss_mask])))
                    X_filled_sf = SimpleFill().fit_transform(x_test_alldata)
                    logger.info("{}--SimpleFillmean:{}".format(modelName, MSE(X_filled_sf[miss_mask], iris[miss_mask])))
                    X_filled_sf_revised = inp.revise(X_filled_sf[miss_mask], miss_location)
                    logger.info(
                        "{}--SimpleFillmean--revised:{}".format(modelName, MSE(X_filled_sf_revised, iris[miss_mask])))
                    logger.info(
                        "{}--SimpleFillmean--improved:{}".format(modelName, MSE(X_filled_sf[miss_mask],
                                                                                         iris[miss_mask]) - MSE(
                            X_filled_sf_revised, iris[miss_mask])))

                    X_filled_md = SimpleFill("median").fit_transform(x_test_alldata)
                    logger.info("{}--SimpleFillmedian:{}".format(modelName, MSE(X_filled_md[miss_mask], iris[miss_mask])))
                    X_filled_md_revised = inp.revise(X_filled_sf[miss_mask], miss_location)
                    logger.info(
                        "{}--SimpleFillmedian--revised:{}".format(modelName, MSE(X_filled_md_revised, iris[miss_mask])))
                    logger.info(
                        "{}--SimpleFillmedian--improved:{}".format(modelName, MSE(X_filled_md[miss_mask],
                                                                                           iris[miss_mask]) - MSE(
                            X_filled_md_revised, iris[miss_mask])))
            except Exception as e:
                logger.error(e)