"""
@author: liuAmon
@contact:utopfish@163.com
@file: auto_encoder.py
@time: 2020/7/14 22:59
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler,SimpleFill
from logger import logger
from Config import baseInfo as bi

from utils.handlerMissData import *
from torch.utils import data

class MyDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        pass
    def __getitem__(self, index):
        pass
    def __len__(self):
        return 0
class Encoder(torch.nn.Module):
    def __init__(self,D_in,theta):
        super(Encoder,self).__init__()
        self.encoder=nn.Sequential(nn.Linear(D_in,D_in+theta),
                                   nn.ReLU(True),
                                   nn.Linear(D_in+theta,D_in+2*theta),
                                   nn.ReLU(True),
                                   nn.Linear(D_in+2*theta,D_in+3*theta))
        # self.linear1 = nn.Linear(D_in, D_out)
        if bi['core']=='cuda':
            self.encoder=self.encoder.cuda()
    def forward(self, x):
        if bi['core'] == 'cuda':
            return F.relu(self.encoder(x)).cuda()
        return F.relu(self.encoder(x))
class Decoder(torch.nn.Module):
    def __init__(self,D_in,theta):
        super(Decoder,self).__init__()
        self.encoder = nn.Sequential(nn.Linear(D_in + 3 * theta, D_in + 2*theta),
                                     nn.ReLU(True),
                                     nn.Linear(D_in + 2*theta, D_in + 1 * theta),
                                     nn.ReLU(True),
                                     nn.Linear(D_in + 1 * theta, D_in ))
        # self.linear1 = nn.Linear(D_in, D_out)
        if bi['core'] == 'cuda':
            self.encoder=self.encoder.cuda()
    def forward(self, x):
        if bi['core'] == 'cuda':
            return F.relu(self.encoder(x)).cuda()
        return F.relu(self.encoder(x))
class AE(nn.Module):
    def __init__(self,encoder,decoder):
        super(AE,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
    def forward(self, state):
        h=self.encoder(state)
        return self.decoder(h)
class interpolation_mask():
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
        self.completeData = completeData
    def fit(self,path,mask):
        if self.feature_length==None:
            self.feature_length=len(self.completeData[0])
            self.h_length=self.feature_length//2

        dataloader = torch.utils.data.DataLoader(self.completeData, batch_size=self.batch_size,
                                                 shuffle=True, num_workers=2)
        encoder = Encoder(self.feature_length, self.h_length)
        decoder = Decoder(self.feature_length, self.h_length)
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
                loss = criterion(dec[mask], inputs[mask])
                loss.backward()
                optimizer.step()
                l = loss.item()
            print(epoch, l)
            if np.isnan(l):
                break
            if epoch % 5==0:
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
        pred_data=X_filled_knn
        device = torch.device(bi['core'])
        encoder = Encoder(self.feature_length, self.h_length)
        decoder = Decoder(self.feature_length, self.h_length)
        ae = AE(encoder, decoder)
        ae.load_state_dict(torch.load(model, map_location=bi['core']))
        criterion = torch.nn.MSELoss()
        fixNum = np.zeros((len(X_filled_knn), len(X_filled_knn[0])))
        inputs = torch.ones(1, self.feature_length, requires_grad=True, device=device)
        for num in range(len(pred_data)):
            for ind in range(len(pred_data[num])):
                inputs.data[0, ind] = pred_data[num][ind]
            for t in range(self.iterations):
                outputs = decoder(encoder(inputs))
                loss = criterion(outputs, inputs)
                loss.backward()
                with torch.no_grad():
                    for mark in miss_mark[num]:
                        inputs.data[0, mark] -= self.learning_rate * inputs.grad.data[0, mark]
                    inputs.grad.zero_()
            fixNum[num] = inputs.data.cpu().numpy()
        return pred_data,fixNum

    def revise(self,beforeRevisedata,miss_loaction,model):
        pred_data = np.array(beforeRevisedata)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(bi['core'])
        encoder = Encoder(self.feature_length, self.h_length)
        decoder = Decoder(self.feature_length, self.h_length)
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
                    for mark in miss_loaction[num]:
                        inputs.data[0, mark] -= self.learning_rate * inputs.grad.data[0, mark]
                    inputs.grad.zero_()
            fixNum[num] = inputs.data.cpu().numpy()
        return fixNum
def get_miss_location(missData):
    miss_location=[]
    for i in missData:
        temp = []
        for index, j in enumerate(i):
            if np.isnan(j):
                temp.append(index)
        miss_location.append(temp)
    return miss_location
def get_mask(missData):
    '''
    对缺失数据缺失部分进行mask
    :param missData:
    :return:
    '''
    mask = np.ones(missData.shape, dtype=bool)
    for row in range(len(missData)):
        for col in range(len(missData[0])):
            if np.isnan(missData[row][col]):
                mask[row][col]=False
                pass
    return mask
def modifier (data,unqiue):
    data=data.copy()
    for i in range(len(data)):
        for j in range(len(data[0])):
                t = np.argmin(np.abs(unqiue-data[i][j]))
                data[i][j] = unqiue[t]
def restore(min_max_scaler,s,data):
    '''
    数据数据反归一化，对插补的数据获取标准插补值，并转为int型
    :return:
    '''
    min_max_scaler.inverse_transform(data)
    data = modifier(data, s)
    return data.astype(int)

def shear_dile(src, dst):
    if os.path.isdir(src):
        if not os.listdir(src):
            os.rmdir(src)
            print('移除空目录: ' + src)
        else:
            for d in os.listdir(src):
                shear_dile(os.path.join(src, d), dst)
    if os.path.isfile(src):
        print("文件剪切:", src)
        fn = os.path.basename(src)
        if not os.path.exists(dst + './' + fn):
            os.rename(src, dst + './' + fn)