#@Time:2020/8/7 19:38
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:tai_test.py
__author__ = "liuAmon"

#基础包
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data
from pandas import isnull
from functools import partial
from logger import logger
from sklearn.preprocessing import StandardScaler
#继承类和model
from utils.tools import Solver
from dnn.autoencoder_test_partice import Autoencoder,ResAutoencoder,StockedAutoencoder,StockedResAutoencoder
from utils.normalizer import NORMALIZERS,RECOVER


#baseline插补方法
from ycimpute.imputer import  mice
from utils.base_impute import random_inpute
from fancyimpute import IterativeImputer, SimpleFill
imputation = {'median':SimpleFill("median").fit_transform,'random':random_inpute,'mice':mice.MICE().complete,'ii':IterativeImputer().fit_transform}
AUTOENCODER_METHOD={'Autoencoder':Autoencoder,'ResAutoencoder':ResAutoencoder,'StockedAutoencoder':StockedAutoencoder,'StockedResAutoencoder':StockedResAutoencoder}
LOSS={'MSELoss':torch.nn.MSELoss(),'CrossEntropyLoss':torch.nn.CrossEntropyLoss()}


class TAI(Solver):
    #原始参数
    def __init__(
            self,
            theta=5,
            epochs=50,
            use_cuda=False,
            batch_size=64,
            early_stop=1e-06,
            normalizer='zero_score',
            iterations=30,
            first_imputation_method='ii',
            learning_rate=0.001,
            loss='MSELoss',
            firstImputedData='None',
            Autoencoder_method='Autoencoder',
            verbose=True):
        Solver.__init__(
            self,
            normalizer=normalizer)

        self.theta = theta
        self.epochs = epochs
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.verbose = verbose
        self.iterations=iterations
        self.early_stop = early_stop
        self.learning_rate=learning_rate
        self.loss=loss
        self.firstImputedData=firstImputedData
        self.Autoencoder_method=Autoencoder_method
        self.first_imputation_method=first_imputation_method

        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

    def training(self, training_data,missing_mask):
        n_features = training_data.shape[1]
        training_mask=1-missing_mask
        training_data = torch.from_numpy(training_data).float()
        train_loader = torch.utils.data.DataLoader(dataset=training_data,
                                                   batch_size=self.batch_size,
                                                   shuffle=False)
        try:
            model_method=partial(AUTOENCODER_METHOD[self.Autoencoder_method])
            model= model_method(dim = n_features,theta = self.theta)
            model=model.to(self.device)
        except Exception as e:
            logger.error(e)
            raise  Exception("please input the correct autoencoder methoed name")
        criterion = LOSS[self.loss]
        optimizer = torch.optim.Adam(model.parameters(),lr=self.learning_rate)
        cost_list = []
        early_stop = False
        for epoch in range(self.epochs):
            for batch_idx, batch_data in enumerate(train_loader):
                inputs = batch_data.to(self.device)
                inputs = inputs.resize_(self.batch_size, n_features)
                dec = model(inputs)
                loss = criterion(dec[training_mask], inputs[training_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # early stopping rule 1 : MSE < 1e-06
                if loss.item() < 1e-06:
                    early_stop = True
                    break
                if batch_idx % 50 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(batch_data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))
                    pass

                cost_list.append(loss.item())

            if early_stop:
                break
        return model
    def get_missing_loc(self,missdata):
        miss_mark = []  # 记录缺失位置
        for i in missdata:
            temp = []
            for index, j in enumerate(i):
                if np.isnan(j):
                    temp.append(index)
            miss_mark.append(temp)
        return miss_mark
    def revise(self,data,model,missing_mask):
        criterion = LOSS[self.loss]

        inputs=torch.tensor(data,dtype=torch.float32, requires_grad=True, device=self.device)


        for t in range(self.iterations):
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            with torch.no_grad():
                #inputs.data -=  inputs.grad.data
                temp=data *(1-missing_mask)+(inputs.data.cpu().numpy()-inputs.grad.data.cpu().numpy())*missing_mask
                inputs =torch.tensor(temp,dtype=torch.float32, requires_grad=True, device=self.device)

                #inputs.grad.zero_()
        final_num = data*(1-missing_mask)+inputs.data.cpu().numpy()*missing_mask
        return final_num
    def solve(self, X,missing_mask):
        model = self.training(self.fill(X.copy(),missing_mask,'zero'),missing_mask)
        model.eval()
        if self.first_imputation_method=="None":
            pred_data =self.fill(X.copy(), missing_mask, 'zero')
        elif self.firstImputedData!='None':
            pred_data= self.ss.transform(self.firstImputedData)
        else:
            pred_data = imputation[self.first_imputation_method](X)
        filled_data =self.revise(data=pred_data.copy(),model=model,missing_mask=missing_mask)
        return filled_data, pred_data

    def complete(self, x):
        """
        Expects 2d float matrix with NaN entries signifying missing values

        Returns completed matrix without any NaNs.
        """
        self._check_input(x)
        try:
            self._check_missing_value_mask(isnull(x))
        #如果输入不含缺失直接返回原数据
        except:
            return x,x
        x, missing_mask = self.prepare_input_data(x)

        x_zero_replaced = self.fill(x.copy(),missing_mask,'zero')
        if self.normalizer is not None:
            self.ss = StandardScaler()
            x_zero_replaced=self.ss.fit_transform(x_zero_replaced)
            # normalizer = NORMALIZERS[self.normalizer]
            # x_zero_replaced, min_record, max_record = normalizer(x_zero_replaced)
        x_zero_replaced[missing_mask]=np.nan
        x_filled,first_filled = self.solve(x_zero_replaced,missing_mask)
        x_filled = self.ss.inverse_transform(x_filled)
        first_filled= self.ss.inverse_transform(first_filled)
        return x_filled,first_filled

if __name__=="__main__":
    pass



