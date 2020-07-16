"""
@author: liuAmon
@contact:utopfish@163.com
@file: test_yci.py
@time: 2020/7/16 15:40
"""
import numpy as np
import pandas as pd
from logger import logger
from ycimpute.utils import evaluate
from matplotlib import pyplot as plt
from ycimpute.imputer import knnimput,iterforest,mice,EM,MIDA,GAIN
from utils.handle_missingdata import gene_missingdata

path = r'../public_data/1_Iris.xlsx'
data = pd.read_excel(path, sheet_name="dataset")
dt = np.array(data.values)
data = dt.astype('float')
origin_data=data[:,:-1]
target=data[:,-1]
knn_rmse=[]
mice_rmse=[]
em_rmse=[]
for i in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    miss_data = gene_missingdata(rate=i, data=origin_data)
    imputed_data=knnimput.KNN(k=3).complete(miss_data)
    score = evaluate.RMSE(origin_data, imputed_data)
    knn_rmse.append(score)
    logger.info("knn missing rate:{},RMSE:{}".format(i,score))
    imputed_data=mice.MICE().complete(miss_data)
    score = evaluate.RMSE(origin_data, imputed_data)
    mice_rmse.append(score)
    logger.info("MICE missing rate:{},RMSE:{}".format(i, score))

    imputed_data=EM().complete(miss_data)
    score = evaluate.RMSE(origin_data, imputed_data)
    em_rmse.append(score)
    logger.info("EM missing rate:{},RMSE:{}".format(i, score))

    # missForest有bug IndexError: too many indices for array
    # imputed_data=iterforest.MissForest().complete(miss_data)
    # score = evaluate.RMSE(origin_data, imputed_data)
    # logger.info("iterforest missing rate:{},RMSE:{}".format(i, score))

    #MIDA has bug ZeroDivisionError: integer division or modulo by zero
    # imputed_data=MIDA().complete(miss_data)
    # score = evaluate.RMSE(origin_data, imputed_data)
    # logger.info("MIDA missing rate:{},RMSE:{}".format(i, score))
    #RuntimeError: DataLoader worker (pid(s) 8180) exited unexpectedly
    # imputed_data=GAIN().complete(miss_data)
    # score = evaluate.RMSE(origin_data, imputed_data)
    # logger.info("GAIN missing rate:{},RMSE:{}".format(i, score))
color=['blue','green','red','yellow','black','burlywood','cadetblue','chartreuse','chocolate','coral']

plt.figure()
x=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

plt.plot(x,knn_rmse,color=color[0],label='knn')
plt.plot(x,mice_rmse,color=color[1],label='mice')
plt.plot(x,em_rmse,color=color[2],label='em')
plt.title("rmse of different missing rate in iris")
plt.legend(loc="upper left")
plt.show()