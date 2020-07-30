"""
@author: liuAmon
@contact:utopfish@163.com
@file: tai.py
@time: 2020/7/23 17:16
"""
from utils.tools import Solver
from dnn.autoencoder import Autoencoder

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from pandas import isnull

from utils.normalizer import NORMALIZERS,RECOVER

from ycimpute.imputer import  mice
from utils.base_impute import random_inpute
from fancyimpute import IterativeImputer, SimpleFill
imputation = {'median':SimpleFill("median").fit_transform,'random':random_inpute,'mice':mice.MICE().complete,'ii':IterativeImputer().fit_transform}
class TAI(Solver):
    def __init__(
            self,
            theta=5,
            epochs=300,
            use_cuda=False,
            batch_size=64,
            early_stop=1e-06,
            normalizer='min_max',
            iterations=20,
            first_imputation_method='ii',
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
        self.first_imputation_method=first_imputation_method

        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

    def training(self, training_data):
        n_features = training_data.shape[1]
        x, missing_mask = self.prepare_input_data(training_data)
        training_mask=1-missing_mask
        training_data = torch.from_numpy(training_data).float()

        train_loader = torch.utils.data.DataLoader(dataset=training_data,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        model = Autoencoder(dim=n_features,
                            theta=self.theta).to(self.device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
        l = None
        bestLoss = np.inf
        cost_list = []
        early_stop = False
        total_batch = len(training_data) // self.batch_size

        for epoch in range(self.epochs):
            for i, batch_data in enumerate(train_loader):
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
    def revise(self,data,model):
        criterion = torch.nn.MSELoss()
        fixNum = np.zeros(data.shape)
        miss_loaction=self.get_missing_loc(data)

        n_feature=data.shape[1]
        inputs = torch.ones(1,n_feature, requires_grad=True, device=self.device)
        for num in range(len(data)):
            for ind in range(len(data[num])):
                inputs.data[0, ind] = data[num][ind]
            for t in range(self.iterations):
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                with torch.no_grad():
                    for mark in miss_loaction[num]:
                        inputs.data[0, mark] -= self.learning_rate * inputs.grad.data[0, mark]
                    # for mark in miss_loaction[num]:
                    #     inputs.data -= self.learning_rate * inputs.grad.data
                    inputs.grad.zero_()
            fixNum[num] = inputs.data.cpu().numpy()
        return fixNum
    def solve(self, X, missing_mask):
        complete_rows_index, missing_rows_index = self.detect_complete_part(missing_mask)
        # if len(complete_rows_index)==0:
        #     raise ValueError('Cant find a completely part for training...')
        # missing_data = X[missing_rows_index]
        # training_data = X[complete_rows_index]

        model = self.training(X.copy())
        model.eval()
        pred_data = imputation[self.first_imputation_method](X)
        # missing_data = torch.from_numpy(X).float()
        filled_data =self.revise(data=pred_data.copy(),model=model)
        # filled_data = model(missing_data.to(self.device))
        # filled_data = filled_data.cpu().detach().numpy()
        # # tmp_mask = missing_mask[missing_rows_index]
        # missing_data = missing_data.cpu().numpy()
        # filled_data = X * (1 - missing_mask) + filled_data * (missing_mask)

        # X[missing_rows_index] = filled_data
        # X[complete_rows_index] = training_data

        return filled_data, pred_data

    def complete(self, x):
        """
        Expects 2d float matrix with NaN entries signifying missing values

        Returns completed matrix without any NaNs.
        """
        self._check_input(x)
        self._check_missing_value_mask(isnull(x))
        x, missing_mask = self.prepare_input_data(x)

        x_zero_replaced = self.fill(x.copy(),missing_mask,'zero')
        if self.normalizer is not None:
            normalizer = NORMALIZERS[self.normalizer]
            x_zero_replaced, min_record, max_record = normalizer(x_zero_replaced)
        x_zero_replaced[missing_mask]=np.nan
        x_filled,first_filled = self.solve(x_zero_replaced, missing_mask)
        revocer = RECOVER[self.normalizer]
        x_filled = revocer(x_filled, min_record, max_record)
        first_filled= revocer(first_filled, min_record, max_record)
        return x_filled,first_filled