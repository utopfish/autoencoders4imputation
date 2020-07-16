#@Time:2019/12/27 20:19
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:examperiment_pub_iris.py
__author__ = "liuAmon"
import os
import time
import numpy as np
import pandas as pd
from logger import logger
from utils.handler_loss import MSE
from sklearn import preprocessing
from utils.base_impute import random_inpute
from auto_encoder import interpolation_mask
from utils.handle_missingdata import gene_missingdata
from utils.base_tools import get_mask,get_miss_location
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler,SimpleFill

if __name__=="__main__":
    start=time.time()

    # #归一化，去掉标签
    path = r'public_data/1_Iris.xlsx'
    fileSavePath = r'G:\labWork\imputed_experiment_data\fix'
    # file='AhyongOM04crabs'

    modelSavePath = r'G:\labWork\imputed_experiment_data\model'
    for file in os.listdir(path):
        logger.info("**********************{}********************".format(file))
        data = pd.read_excel(path, sheet_name="dataset")
        dt = np.array(data.values)
        data = dt.astype('float')
        data=data[:-1]
        target=data[-1]
        for i in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

            miss_data = gene_missingdata(rate=i, data=data)

            mask = get_mask(miss_data)
            miss_location = get_miss_location(miss_data)
            # 数据均值化
            # min_max_scaler = preprocessing.MinMaxScaler()
            # mm_miss_data = min_max_scaler.fit_transform(miss_data)
            min_max_scaler = preprocessing.StandardScaler()
            mm_miss_data = min_max_scaler.fit_transform(miss_data)
            modelName = file + str(i)
            inp = interpolation_mask(modelName=modelName, completeData=random_inpute(mm_miss_data))
            if not os.path.exists(os.path.join(modelSavePath, '{}.pkl'.format(modelName))):
                inp.fit(os.path.join(modelSavePath, '{}.pkl'.format(modelName)), mask)

            # pred_data, fix_data = inp.prd(mm_miss_data, model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
            #
            # fix_auto_data = min_max_scaler.inverse_transform(fix_data)
            #
            # pred_data = min_max_scaler.inverse_transform(pred_data)
            # # saveData(fileSavePath, '{}_{}_auto.tnt'.format(file, str(i / 2)), speciesname, fix_auto_data, begin, end)
            # # logger.info("{}============".format(i))
            # # logger.info("number of charater :{}".format(len(s)))
            # logger.info("初步插补 MSE:{}".format(MSE(data, pred_data)))
            # logger.info("ae MSE:{}".format(MSE(data,fix_auto_data)))
            # logger.info("res change MSE:{}".format(MSE(pred_data, fix_auto_data)))


            X_filled_knn = KNN(k=3).fit_transform(mm_miss_data)
            re_X = inp.revise(X_filled_knn, miss_location,
                              model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
            X_filled_knn = min_max_scaler.inverse_transform(X_filled_knn)
            re_X = min_max_scaler.inverse_transform(re_X)
            logger.info("knn MSE:{}".format(MSE(data, X_filled_knn)))
            logger.info("knn res MSE:{}".format(MSE(data, re_X)))
            logger.info("res change MSE:{}".format(MSE(X_filled_knn, re_X)))
            # X_filled_ii = IterativeImputer().fit_transform(mm_miss_data)
            # re_X = inp.revise(X_filled_ii, miss_location,
            #                   model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
            # X_filled_ii = restore(min_max_scaler=min_max_scaler,s=s,data=X_filled_ii)
            # re_X = restore(min_max_scaler=min_max_scaler, s=s, data=re_X)
            # logger.info("ii MSE:{}".format(MSE(imputed_data, X_filled_ii)))
            # logger.info("ii res MSE:{}".format(MSE(imputed_data,  re_X)))

            X_filled_sf = SimpleFill().fit_transform(mm_miss_data)
            re_X = inp.revise(X_filled_sf, miss_location,
                              model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
            X_filled_sf = min_max_scaler.inverse_transform(X_filled_sf)
            re_X = min_max_scaler.inverse_transform(re_X)
            logger.info("sf MSE:{}".format(MSE(data, X_filled_sf)))
            logger.info("sf res MSE:{}".format(MSE(data, re_X)))
            logger.info("res change MSE:{}".format(MSE(X_filled_sf, re_X)))
            X_filled_me = SimpleFill("median").fit_transform(miss_data)
            re_X = inp.revise(X_filled_me, miss_location,
                              model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
            X_filled_me = min_max_scaler.inverse_transform(X_filled_me)
            re_X = min_max_scaler.inverse_transform(re_X)
            logger.info("median MSE:{}".format(MSE(data, X_filled_me)))
            logger.info("median res MSE:{}".format(MSE(data,re_X)))
            logger.info("res change MSE:{}".format(MSE(X_filled_me, re_X)))

            # result[e]=temp
        # logger.info(result)

        logger.info(" missing rate is {}".format(i))

        break
