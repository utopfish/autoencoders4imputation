# """
# @author: liuAmon
# @contact:utopfish@163.com
# @file: test_baseline_imputation_true.py
# @time: 2020/7/22 20:04
# """
# import os
# import impyute
# import numpy as np
# import pandas as pd
#
# from logger import logger
# from ycimpute.utils import evaluate
# from matplotlib import pyplot as plt
# from utils.handle_missingdata import gene_missingdata,gene_missingdata_taxa_bias,gene_missingdata_chara_bias,gene_missingdata_block_bias
# from dnn.mida import MIDA
# from dnn.gain import GAIN
# from dnn.tai import TAI
# from ycimpute.imputer import knnimput, mice, EM
# from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler, SimpleFill
#
# path = r'../public_data/'
# pciturePath = r'G:\labWork\imputation_plt\pub_my_method'
# for file in os.listdir(path):
#     logger.info("**********************{}********************".format(file))
#     data = pd.read_excel(os.path.join(path, file), sheet_name="dataset")
#     dt = np.array(data.values)
#     data = dt.astype('float')
#     origin_data = data[:, :-1]
#     target = data[:, -1]
#
#     for miss_pat in ['normal','taxa','chara','block']:
#         mice_rmse = []
#         ii_rmse = []
#         median_rmse = []
#         random_rmse = []
#         mida_rmse = []
#         gain_rmse = []
#         tai_rmse=[]
#         for i in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#             if miss_pat=='normal':
#                 miss_data = gene_missingdata(rate=i, data=origin_data)
#             elif miss_pat=='taxa':
#                 miss_data =gene_missingdata_taxa_bias(rate=i, data=origin_data)
#             elif miss_pat =='chara':
#                 miss_data =gene_missingdata_chara_bias(rate=i, data=origin_data)
#             elif miss_pat== 'block':
#                 miss_data=gene_missingdata_block_bias(rate=i, data=origin_data)
#             else:
#                 raise Exception("缺失模式错误，请在'normal','taxa','chara','block'中选择对应模式")
#             try:
#                 imputed_data = mice.MICE().complete(miss_data)
#                 score = evaluate.RMSE(origin_data, imputed_data)
#                 mice_rmse.append(score)
#                 logger.info("MICE missing rate:{},RMSE:{}".format(i, score))
#             except:
#                 mice_rmse.append(np.inf)
#             try:
#                 imputed_data = IterativeImputer().fit_transform(miss_data)
#                 score = evaluate.RMSE(origin_data, imputed_data)
#                 ii_rmse.append(score)
#                 logger.info("fi IterativeImputer missing rate:{},RMSE:{}".format(i, score))
#             except:
#                 ii_rmse.append(np.inf)
#             try:
#                 imputed_data = SimpleFill("median").fit_transform(miss_data)
#                 score = evaluate.RMSE(origin_data, imputed_data)
#                 median_rmse.append(score)
#                 logger.info("fi median missing rate:{},RMSE:{}".format(i, score))
#             except:
#                 median_rmse.append(np.inf)
#             try:
#                 imputed_data = impyute.imputation.cs.random(miss_data)
#                 score = evaluate.RMSE(origin_data, imputed_data)
#                 random_rmse.append(score)
#                 logger.info("random missing rate:{},RMSE:{}".format(i, score))
#             except:
#                 random_rmse.append(np.inf)
#             try:
#                 imputed_data = MIDA().complete(miss_data)
#                 score = evaluate.RMSE(origin_data, imputed_data)
#                 logger.info("MIDA missing rate:{},RMSE:{}".format(i, score))
#                 mida_rmse.append(score)
#             except:
#                 mida_rmse.append(np.inf)
#             try:
#                 imputed_data = GAIN().complete(miss_data)
#                 score = evaluate.RMSE(origin_data, imputed_data)
#                 logger.info("GAIN missing rate:{},RMSE:{}".format(i, score))
#                 gain_rmse.append(score)
#             except:
#                 gain_rmse.append(np.inf)
#
#             try:
#                 imputed_data = TAI().complete(miss_data)
#                 score = evaluate.RMSE(origin_data, imputed_data)
#                 logger.info("TAI missing rate:{},RMSE:{}".format(i, score))
#                 tai_rmse.append(score)
#             except:
#                 tai_rmse.append(np.inf)
#
#         color = ['blue', 'green', 'red', 'yellow', 'black', 'burlywood', 'cadetblue', 'chartreuse', 'purple', 'coral',
#                  'aqua', 'aquamarine', 'darkblue', 'y','m']
#
#         x = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#         plt.plot(x, mice_rmse, color=color[1], label='mice')
#         plt.plot(x, ii_rmse, color=color[2], label='ii')
#         plt.plot(x, median_rmse, color=color[3], linestyle=':', label='median')
#         plt.plot(x, random_rmse, color=color[9], linestyle=':', label='random')
#         plt.plot(x, mida_rmse, color=color[13], linewidth=3.0, linestyle='-.', label='mida')
#         plt.plot(x, gain_rmse, color=color[14], linewidth=3.0, linestyle='-.', label='gain')
#         plt.plot(x, tai_rmse, color=color[10], linewidth=3.0, linestyle='-', label='tai')
#         plt.title("rmse of different missing rate in {}_{}".format(file, miss_pat))
#         plt.legend(loc="upper left")
#         # plt.show()
#         plt.savefig(os.path.join(pciturePath, "rmse_of_different_missing_rate_in_{}_{}.png".format(file, miss_pat)))
