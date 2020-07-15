#@Time:2019/12/26 19:32
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:fancyimpute_test.py
__author__ = "liuAmon"
import numpy as np
import pandas as pd
from utils.handlerMissData import geneMissData
from utils.handlerLoss import MSE
from sklearn import preprocessing
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler,SimpleFill

iris = pd.read_excel('../public_data/1_Iris.xlsx', sheet_name="dataset")
dt = np.array(iris.values)
iris = dt.astype('float')
# #归一化，去掉标签
min_max_scaler = preprocessing.MinMaxScaler()
iris = min_max_scaler.fit_transform(iris[:, :-1])
x_test_alldata,miss_mask = geneMissData(0.2, iris)

X_filled_knn=KNN(k=3).fit_transform(x_test_alldata)
#X_filled_Simple=SoftImpute(x_test_alldata)
print("knn:{}".format(MSE(X_filled_knn[miss_mask],iris[miss_mask])))
#print(MSE(X_filled_Simple,iris))

# X_incomplete_normalized = BiScaler().fit_transform(x_test_alldata)
# X_filled_softimpute = SoftImpute().fit_transform(X_incomplete_normalized)
# print("softimpute:{}".format(MSE(X_filled_softimpute,iris)))

X_filled_ii = IterativeImputer().fit_transform(x_test_alldata)
print("iterativeImputer:{}".format(MSE(X_filled_ii[miss_mask],iris[miss_mask])))

X_filled_sf=SimpleFill().fit_transform(x_test_alldata)
print("SimpleFillmean:{}".format(MSE(X_filled_sf[miss_mask],iris[miss_mask])))
X_filled_sf=SimpleFill("median").fit_transform(x_test_alldata)
print("SimpleFillmedian:{}".format(MSE(X_filled_sf[miss_mask],iris[miss_mask])))
# X_filled_nnm = NuclearNormMinimization().fit_transform(x_test_alldata)
# print("NuclearNorm:{}".format(MSE(X_filled_nnm,iris)))

# X_incomplete_normalized = BiScaler().fit_transform(x_test_alldata)
# X_filled_softimpute = SoftImpute().fit_transform(X_incomplete_normalized)
# print("softimpute:{}".format(MSE(X_filled_softimpute,iris)))

