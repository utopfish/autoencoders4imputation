#@Time:2019/12/25 16:17
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:iris_experiment_revise.py
__author__ = "liuAmon"
import time
import torch
import numpy as np
import pandas as pd
from utils import kdtree
from logger import logger
from utils.handler_loss import MSE
from sklearn import preprocessing
from Config import baseInfo as bi
from utils.kmeans import KMeansBase
from utils.handle_missingdata import gene_missingdata
from sklearn.model_selection import train_test_split
from autoEncoder_pytorch import AE,Encoder,Decoder



def test(number,data):
    x_test_alldata, x_test_complete, x_test_miss, x_test_miss_data_true = gene_missingdata(0.2, data)
    km = KMeansBase(3, distanceType="euclidean")
    labels = km.fit_predict(x_test_complete)
    print(km.centers)
    data = [tuple(i) for i in km.centers]
    tree = kdtree.create(data)
    pred_data = []  # 预测数据
    miss_mark = []  # 记录缺失位置
    for i in x_test_miss:
        temp = []
        for index, j in enumerate(i):
            if np.isnan(j):
                temp.append(index)
        miss_mark.append(temp)
    for ind, i in enumerate(x_test_miss):
        t = tree.search_knn(i, 1, distanceType='euclidean')
        closeNode = list(t[0][0].data)
        interped_dat = list(i.copy())
        for ind in range(len(interped_dat)):
            if np.isnan(interped_dat[ind]):
                interped_dat[ind] = closeNode[ind]
        pred_data.append(interped_dat)
    pred_data = np.array(pred_data)

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(bi['core'])
    encoder = Encoder(feature_length, ae_length)
    decoder = Decoder(ae_length, feature_length)
    ae = AE(encoder, decoder)
    ae.load_state_dict(torch.load('iris.pkl',map_location=bi['core']))
    criterion = torch.nn.MSELoss()
    fixNum = np.zeros((len(pred_data), len(pred_data[0])))
    inputs = torch.ones(1, feature_length, requires_grad=True, device=device)
    for num in range(len(pred_data)):
        for ind in range(len(pred_data[num])):
            inputs.data[0, ind] = pred_data[num][ind]
        for t in range(iterations):
            outputs = decoder(encoder(inputs))
            loss = criterion(outputs, inputs)
            loss.backward()
            # if t%20==0:
            #     print("input:{}".format(inputs))
            #     print("inputs grad：{}".format(inputs.grad))
            #     print("loss:{}".format(loss))
            with torch.no_grad():
                for mark in miss_mark[num]:
                    inputs.data[0, mark] -= learning_rate * inputs.grad.data[0, mark]
                inputs.grad.zero_()
        t = inputs.data
        fixNum[num] = inputs.data.cpu().numpy()
    beforeRevise=MSE(pred_data, x_test_miss_data_true)
    afterRevise=MSE(fixNum, x_test_miss_data_true)
    logger.info("no {} experiment MSE increase:{}".format(number,beforeRevise-afterRevise))
    logger.info("before revise:{}".format(beforeRevise))
    logger.info("after revise:{}".format(afterRevise))

if __name__=="__main__":
    feature_length = 4  # 特征维度
    ae_length = 2  # 自编码器压缩维度
    learning_rate = 1  # 自编码器数据校正学习率
    iterations = 100  # 自编码器数据校正迭代次数
    iris = pd.read_excel('public_data/1_Iris.xlsx', sheet_name="dataset")
    dt = np.array(iris.values)
    iris = dt.astype('float')
    # #归一化，去掉标签
    min_max_scaler = preprocessing.MinMaxScaler()
    iris = min_max_scaler.fit_transform(iris[:, :-1])
    x_train, x_test = train_test_split(iris, test_size=0.3, random_state=0)
    for i in range(10000):
        start=time.time()
        try:
            test(i,iris)
        except Exception as e :
            logger.error(e)
        print("耗时{}".format(time.time()-start))