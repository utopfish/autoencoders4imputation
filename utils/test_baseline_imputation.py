"""
@author: liuAmon
@contact:utopfish@163.com
@file: test_baseline_imputation.py
@time: 2020/7/16 16:53
"""
import numpy as np
import pandas as pd
from logger import logger
from ycimpute.utils import evaluate
from matplotlib import pyplot as plt
from ycimpute.imputer import knnimput,iterforest,mice,EM,MIDA,GAIN
from utils.handle_missingdata import gene_missingdata
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler,SimpleFill
path = r'../public_data/1_Iris.xlsx'
data = pd.read_excel(path, sheet_name="dataset")
dt = np.array(data.values)
data = dt.astype('float')
origin_data=data[:,:-1]
target=data[:,-1]
knn_rmse=[]
mice_rmse=[]
em_rmse=[]
fi_knn_rmse=[]
fi_bs_rmse=[]
fi_si_rmse=[]
fi_ii_rmse=[]
fi_sf_rmse=[]
fi_median_rmse=[]

for i in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    miss_data = gene_missingdata(rate=i, data=origin_data)
    try:
        imputed_data=knnimput.KNN(k=3).complete(miss_data)
        score = evaluate.RMSE(origin_data, imputed_data)
        knn_rmse.append(score)
        logger.info("knn missing rate:{},RMSE:{}".format(i,score))
    except:
        knn_rmse.append(np.nan)
    try:
        imputed_data=mice.MICE().complete(miss_data)
        score = evaluate.RMSE(origin_data, imputed_data)
        mice_rmse.append(score)
        logger.info("MICE missing rate:{},RMSE:{}".format(i, score))
    except:
        mice_rmse.append(np.nan)
    try:
        imputed_data=EM().complete(miss_data)
        score = evaluate.RMSE(origin_data, imputed_data)
        em_rmse.append(score)
        logger.info("EM missing rate:{},RMSE:{}".format(i, score))
    except:
        em_rmse.append(np.nan)
    try:
        imputed_data = KNN(k=3).fit_transform(miss_data)
        score = evaluate.RMSE(origin_data, imputed_data)
        fi_knn_rmse.append(score)
        logger.info("fi knn missing rate:{},RMSE:{}".format(i, score))
    except:
        fi_knn_rmse.append(np.nan)
    try:
        imputed_data = BiScaler().fit_transform(miss_data)
        imputed_data = SoftImpute().fit_transform(imputed_data)
        score = evaluate.RMSE(origin_data, imputed_data)
        fi_bs_rmse.append(score)
        logger.info("fi BiScaler  missing rate:{},RMSE:{}".format(i, score))
    except:
        fi_bs_rmse.append(np.nan)
    try:
        imputed_data= SoftImpute().fit_transform(miss_data)
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

color=['blue','green','red','yellow','black','burlywood','cadetblue','chartreuse','chocolate','coral']

plt.figure()
x=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

plt.plot(x,knn_rmse,color=color[0],label='knn')
plt.plot(x,mice_rmse,color=color[1],label='mice')
plt.plot(x,em_rmse,color=color[2],label='em')
plt.plot(x,fi_knn_rmse,color=color[3],label='fi knn')
plt.plot(x,fi_bs_rmse,color=color[4],label='fi bs')
plt.plot(x,fi_si_rmse,color=color[5],label='fi si')
plt.plot(x,fi_ii_rmse,color=color[6],label='fi ii')
plt.plot(x,fi_sf_rmse,color=color[7],label='fi sf')
plt.plot(x,fi_median_rmse,color=color[8],label='fi median')
plt.title("rmse of different missing rate in iris")
plt.legend(loc="upper left")
plt.show()