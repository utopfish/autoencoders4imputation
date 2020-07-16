from sklearn import preprocessing
import os
from utils.read_file import *
from utils.random_impute import *
from logger import logger
from utils.handler_loss import *
from utils.handle_missingdata import *
import os
import torch
from utils.handler_loss import *
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from torch.autograd import Variable
from utils.random_impute import *
from Config import baseInfo as bi
from logger import logger
from utils import kdtree
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler,SimpleFill
from logger import logger
from Config import baseInfo as bi
from utils.handler_loss import MSE
from sklearn import preprocessing
from torch.autograd import Variable
from utils.kmeans import KMeansBase
from utils.read_file import readNex
from utils.handle_missingdata import *
from autoEncoder_pytorch import get_mask,get_miss_location
class Encoder(torch.nn.Module):
    def __init__(self,D_in,D_out):
        super(Encoder,self).__init__()

        self.linear1 = nn.Linear(D_in, D_out)
        if bi['core']=='cuda':
            self.linear1=self.linear1.cuda()
    def forward(self, x):
        if bi['core'] == 'cuda':
            return F.relu(self.linear1(x)).cuda()
        return F.relu(self.linear1(x))
class Decoder(torch.nn.Module):
    def __init__(self,D_in,D_out):
        super(Decoder,self).__init__()
        self.linear1 = nn.Linear(D_in, D_out)
        if bi['core'] == 'cuda':
            self.linear1=self.linear1.cuda()
    def forward(self, x):
        if bi['core'] == 'cuda':
            return F.relu(self.linear1(x)).cuda()
        return F.relu(self.linear1(x))
class AE(nn.Module):
    def __init__(self,encoder,decoder):
        super(AE,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
    def forward(self, state):
        h=self.encoder(state)
        return self.decoder(h)
class interpolation_mask_onehot():
    '''
    自编码，对缺失部分数据进行均值插补，损失函数将缺失数据掩盖
    '''
    def __init__(self,feature_length=None,h_length=None,modelName=None,completeData=None):
        self.batch_size=90
        self.epochs=20
        self.learning_rate=0.01
        self.iterations=20
        self.feature_length=feature_length
        self.h_length=h_length
        self.modelName=modelName
        self.enc = preprocessing.OneHotEncoder()
        self.enc.fit(completeData)
        self.completeData = self.enc.transform(completeData).toarray()
    def fit(self,path,mask):
        if self.feature_length==None:
            self.feature_length=len(self.completeData[0])
            self.h_length=self.feature_length//2

        dataloader = torch.utils.data.DataLoader(self.completeData, batch_size=self.batch_size,
                                                 shuffle=True, num_workers=2)
        encoder = Encoder(self.feature_length, self.h_length)
        decoder = Decoder(self.h_length, self.feature_length)
        ae = AE(encoder, decoder)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(ae.parameters(), lr=self.learning_rate)
        l = None
        bestLoss = np.inf
        for epoch in range(self.epochs):
            for data in dataloader:
                inputs = data.float()
                inputs = inputs.resize_(self.batch_size, self.feature_length)
                if bi['core'] == 'cuda':
                    inputs = inputs.cuda()
                optimizer.zero_grad()
                dec = ae(inputs)

                dec_c=self.enc.inverse_transform(dec.data.cpu().numpy())
                inputs_c=self.enc.inverse_transform(inputs.data.cpu().numpy())
                t=dec[mask]
                k=inputs[mask]
                dec=torch.from_numpy(dec_c[mask])
                inputs=torch.from_numpy(inputs_c[mask])
                loss = criterion(dec, inputs)
                loss.backward()
                optimizer.step()
                l = loss.item()
            print(epoch, l)
            if np.isnan(l):
                break
            if l < bestLoss:
                bestLoss = l
                torch.save(ae.state_dict(), path)
                logger.info("{}'s loss is {}".format(path,l))
                print("模型保存成功")
    def prd(self,missdata,model):
        '''
        换用knn进行预插补,并进行缺失矫正
        :param missdata:
        :param model:
        :return:
        '''
        if self.feature_length==None:
            self.feature_length=len(self.completeData[0])
            self.h_length=self.feature_length//2
        miss_mark = []  # 记录缺失位置
        for i in missdata:
            temp = []
            for index, j in enumerate(i):
                if np.isnan(j):
                    temp.append(index)
            miss_mark.append(temp)
        X_filled_knn = KNN(k=3).fit_transform(missdata)
        s = pd.DataFrame(np.unique(missdata)).dropna(axis=0).values
        s = [i[0] for i in s]
        X_filled_knn=modifier(X_filled_knn,s)
        pred_data=self.enc.transform(X_filled_knn).toarray()

        device = torch.device(bi['core'])
        encoder = Encoder(self.feature_length, self.h_length)
        decoder = Decoder(self.h_length, self.feature_length)
        ae = AE(encoder, decoder)
        ae.load_state_dict(torch.load(model, map_location=bi['core']))
        criterion = torch.nn.MSELoss()
        fixNum = np.zeros((len(pred_data), len(pred_data[0])))
        inputs = torch.ones(1, self.feature_length, requires_grad=True, device=device)
        for num in range(len(pred_data)):
            for ind in range(len(pred_data[num])):
                inputs.data[0, ind] = pred_data[num][ind]
            for t in range(self.iterations):
                outputs = decoder(encoder(inputs))
                loss = criterion(outputs, inputs)
                loss.backward()
                with torch.no_grad():
                    # for mark in miss_mark[num]:
                    inputs.data[0] -= self.learning_rate * inputs.grad.data[0]
                    inputs.grad.zero_()
            fixNum[num] = inputs.data.cpu().numpy()
        return X_filled_knn,fixNum
    def revise(self,beforeRevisedata,miss_loaction,model):
        pred_data = np.array(beforeRevisedata)
        #加入onehot

        pred_data = self.enc.transform(pred_data).toarray()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(bi['core'])
        encoder = Encoder(self.feature_length, self.h_length)
        decoder = Decoder(self.h_length, self.feature_length)
        ae = AE(encoder, decoder)
        ae.load_state_dict(torch.load(model, map_location=bi['core']))
        criterion = torch.nn.MSELoss()
        fixNum = np.zeros((len(pred_data), len(pred_data[0])))
        inputs = torch.ones(1, self.feature_length, requires_grad=True, device=device)
        for num in range(len(pred_data)):
            for ind in range(len(pred_data[num])):
                inputs.data[0, ind] = pred_data[num][ind]
            for t in range(self.iterations):
                outputs = decoder(encoder(inputs))
                loss = criterion(outputs, inputs)
                loss.backward()
                with torch.no_grad():
                    # for mark in miss_loaction[num]:
                    inputs.data[0] -= self.learning_rate * inputs.grad.data[0]
                    print(np.sum(inputs.data.cpu().numpy()))
                    inputs.grad.zero_()
            fixNum[num] = inputs.data.cpu().numpy()
        fixNum=self.enc.inverse_transform(fixNum)
        return fixNum
if __name__=="__main__":
    from examperiment_ped import modifier
    path=r'G:\labWork\cladistic-data-master\nexus_files'
    fileSavePath = r'G:\labWork\imputed_experiment_data\fix'
    # file='AhyongOM04crabs'

    modelSavePath = r'G:\labWork\imputed_experiment_data\model'
    # for file in os.listdir(path):
    file='AhyongOM04crabs.nex'
    try:
        data, misss_row, speciesname, begin, end = readNex(os.path.join(path,file))
    except ValueError:
        print("可能存在数据多态问题")
        # continue
    # try:
    imputed_data = random_inpute(data)
    saveData(fileSavePath, '{}_init.tnt'.format(file), speciesname, imputed_data, begin, end)

    logger.info("**********************{}********************".format(file))
    for  i in [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        miss_data= gene_missingdata(rate=i, data=imputed_data)

        logger.info("{}============".format(i))
        logger.info("random MSE:{}".format(MSE(imputed_data,random_inpute(miss_data))))

        mask=get_mask(miss_data)
        miss_location=get_miss_location(miss_data)
        #数据均值化
        # min_max_scaler = preprocessing.MinMaxScaler()
        # mm_miss_data = min_max_scaler.fit_transform(miss_data)
        modelName = file + str(i)
        inp = interpolation_mask_onehot(modelName=modelName, completeData=random_inpute(miss_data))
        if not os.path.exists(os.path.join(modelSavePath, '{}.pkl'.format(modelName))):
            inp.fit(os.path.join(modelSavePath, '{}.pkl'.format(modelName)),mask)

        pred_data, fix_data = inp.prd(miss_data, model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))

        s = pd.DataFrame(np.unique(pred_data)).dropna(axis=0).values
        s=[i[0] for i in s]
        fix_data = modifier(fix_data, s)
        fix_data=fix_data.astype(int)
        fix_data=inp.enc.inverse_transform(fix_data)
        # fix_auto_data = restore(min_max_scaler=min_max_scaler,s=s,data=fix_data)

        # pred_data=restore(min_max_scaler=min_max_scaler,s=s,data=pred_data)
        # saveData(fileSavePath, '{}_{}_auto.tnt'.format(file, str(i / 2)), speciesname, fix_auto_data, begin, end)
        # logger.info("{}============".format(i))
        logger.info("number of charater :{}".format(len(s)))
        pred_data = modifier(pred_data, s)
        pred_data = pred_data.astype(int)
        logger.info("初步插补 MSE:{}".format(MSE(imputed_data, pred_data)))
        logger.info("ae MSE:{}".format(MSE(imputed_data,fix_data)))

        # logger.info("初步插补 TF:{}".format(TF(imputed_data, pred_data)))
        # logger.info("ae TF:{}".format(TF(imputed_data, fix_data)))

        X_filled_knn = KNN(k=3).fit_transform(miss_data)
        re_X = inp.revise(modifier(X_filled_knn, s), miss_location,
                                     model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
        re_X = modifier(re_X, s)
        re_X = re_X.astype(int)
        X_filled_knn = modifier(X_filled_knn, s)
        X_filled_knn = X_filled_knn.astype(int)
        logger.info("knn MSE:{}".format(MSE(imputed_data, X_filled_knn)))
        logger.info("knn res MSE:{}".format(MSE(imputed_data, re_X)))
        logger.info("res  change MSE:{}".format(MSE(X_filled_knn, re_X)))

        # X_filled_ii = IterativeImputer().fit_transform(mm_miss_data)
        # re_X = inp.revise(X_filled_ii, miss_location,
        #                   model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
        # X_filled_ii = restore(min_max_scaler=min_max_scaler,s=s,data=X_filled_ii)
        # re_X = restore(min_max_scaler=min_max_scaler, s=s, data=re_X)
        # logger.info("ii MSE:{}".format(MSE(imputed_data, X_filled_ii)))
        # logger.info("ii res MSE:{}".format(MSE(imputed_data,  re_X)))

        X_filled_sf = SimpleFill().fit_transform(miss_data)
        re_X  = inp.revise(modifier(X_filled_sf, s), miss_location,
                                     model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
        re_X = modifier(re_X, s)
        re_X = re_X.astype(int)
        X_filled_sf = modifier(X_filled_sf, s)
        X_filled_sf = X_filled_sf.astype(int)
        logger.info("sf MSE:{}".format(MSE(imputed_data, X_filled_sf)))
        logger.info("sf res MSE:{}".format(MSE(imputed_data, re_X)))
        logger.info("res  change MSE:{}".format(MSE(X_filled_sf, re_X)))

        X_filled_me = SimpleFill("median").fit_transform(miss_data)
        re_X = inp.revise(modifier(X_filled_me, s), miss_location,
                          model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
        re_X = modifier(re_X, s)
        re_X = re_X.astype(int)
        X_filled_me = modifier(X_filled_me, s)
        X_filled_me =X_filled_me.astype(int)
        logger.info("median MSE:{}".format(MSE(imputed_data, X_filled_me)))
        logger.info("median res MSE:{}".format(MSE(imputed_data,re_X)))
        logger.info("res  change MSE:{}".format(MSE(X_filled_me, re_X)))
    # except Exception as e:
    #     logger.error(e)