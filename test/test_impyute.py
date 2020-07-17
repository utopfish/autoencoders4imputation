"""
@author: liuAmon
@contact:utopfish@163.com
@file: test_impyute.py
@time: 2020/7/17 1:01
"""
import numpy as np
import pandas as pd
from logger import logger
from ycimpute.utils import evaluate
from matplotlib import pyplot as plt
from ycimpute.imputer import knnimput,iterforest,mice,EM,MIDA,GAIN
from utils.handle_missingdata import gene_missingdata
import impyute
path = r'../public_data/1_Iris.xlsx'
data = pd.read_excel(path, sheet_name="dataset")
dt = np.array(data.values)
data = dt.astype('float')
origin_data=data[:,:-1]
target=data[:,-1]
random_rmse=[]
mean_rmse=[]
mode_rmse=[]
median_rmse=[]
for i in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    miss_data = gene_missingdata(rate=i, data=origin_data)
    imputed_data = impyute.imputation.cs.random(miss_data)
    score = evaluate.RMSE(origin_data, imputed_data)
    random_rmse.append(score)
    logger.info("random missing rate:{},RMSE:{}".format(i, score))

    imputed_data = impyute.imputation.cs.mean(miss_data)
    score = evaluate.RMSE(origin_data, imputed_data)
    mean_rmse.append(score)
    logger.info("mean missing rate:{},RMSE:{}".format(i, score))

    imputed_data = impyute.imputation.cs.mode(miss_data)
    score = evaluate.RMSE(origin_data, imputed_data)
    mode_rmse.append(score)
    logger.info("mode missing rate:{},RMSE:{}".format(i, score))

    imputed_data = impyute.imputation.cs.median(miss_data)
    score = evaluate.RMSE(origin_data, imputed_data)
    median_rmse.append(score)
    logger.info("median missing rate:{},RMSE:{}".format(i, score))


plt.figure()
x=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
color=['blue','green','red','yellow','black','burlywood','cadetblue','chartreuse','chocolate','coral']
plt.plot(x,random_rmse,color=color[0],label='random')
plt.plot(x,mean_rmse,color=color[1],label='mean')
plt.plot(x,mode_rmse,color=color[2],label='mode')
plt.plot(x,median_rmse,color=color[3],label='median')
plt.title("rmse of different missing rate in iris")
plt.legend(loc="upper left")
plt.show()