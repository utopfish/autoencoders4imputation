"""
@author: liuAmon
@contact:utopfish@163.com
@file: test_gain.py
@time: 2020/7/16 17:06
"""
import numpy as np

import os


def shulle_data(data):
    seed = 2323  # 1314(second val) #123456(first val)

    np.random.seed(seed)
    np.random.shuffle(data)

    return data

def missing(m, n, rate):
    p_miss_vec = rate * np.ones((n, 1))
    Missing = np.zeros((m, n))

    for i in range(n):
        A = np.random.uniform(0., 1., size=[m, ])
        B = A > p_miss_vec[i]
        Missing[:, i] = 1. * B

    return Missing

def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size = [m, n])


def make_dataset(data_path, missing_rate, train_ratio=0.8, is_label_numerical=False):
    data = np.loadtxt(data_path, delimiter=',')

    data = shulle_data(data)

    label = data[:, 0]
    data = data[:, 1:]

    data_dim = data.shape[1]
    min_val = np.zeros(data_dim)
    max_val = np.zeros(data_dim)
    min_label = None
    max_label = None

    for i in range(data_dim):
        min_val[i] = np.min(data[:, i])
        max_val[i] = np.max(data[:, i])
        if max_val[i] == 0:
            max_val[i] = 0.1

    if is_label_numerical:
        min_label = np.min(label)
        max_label = np.max(label)

        label = (label - min_label) / (max_label - min_label)

    missing_mat = missing(data.shape[0], data.shape[1],
                          missing_rate)

    train = data[:int(train_ratio * data.shape[0])]
    test = data[int(train_ratio * data.shape[0]):]

    train_label = label[:int(train_ratio * data.shape[0])]
    test_label = label[int(train_ratio * data.shape[0]):]

    train_missing = missing_mat[:int(train_ratio * data.shape[0])]
    test_missing = missing_mat[int(train_ratio * data.shape[0]):]

    train_noise = sample_Z(train.shape[0], train.shape[1])
    test_noise = sample_Z(test.shape[0], test.shape[1])

    info = {'train': train,
            'test': test,
            'train_missing': train_missing,
            'test_missing': test_missing,
            'train_noise': train_noise,
            'test_noise': test_noise,
            'min_val': min_val,
            'max_val': max_val,
            'train_label': train_label,
            'test_label': test_label,
            'max_label': max_label,
            'min_label': min_label,
            'missing_rate': missing_rate,
            'train_rate': train_ratio
            }

    return info

dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(dir_path,'test_data/wave.csv')
info_dict = make_dataset(data_path=data_path,
                         missing_rate=0.2,
                         is_label_numerical=False)


missing_mask = info_dict['train_missing'][2000:2600,:]
complete_data = info_dict['train'][2000:2600,:]

missing_mask[:300] = True
missing_data = complete_data.copy()
missing_mask = missing_mask.astype(bool)
missing_mask = ~missing_mask
missing_data[missing_mask]=np.nan
from  ycimpute.imputer import GAIN,MIDA
from ycimpute.utils import evaluate
from ycimpute.utils.normalizer import min_max_scale
def test_gain():
    X_filled = GAIN().complete(missing_data)
    complete_data_, _, _ = min_max_scale(complete_data)
    X_filled, _, _ = min_max_scale(X_filled)

    score = evaluate.RMSE(complete_data_[missing_mask],
                 X_filled[missing_mask])
    print(score)
def test_mida():
    X_filled = MIDA().complete(missing_data)
    complete_data_, _, _ = min_max_scale(complete_data)
    X_filled, _, _ = min_max_scale(X_filled)

    score = evaluate.RMSE(complete_data_[missing_mask],
                 X_filled[missing_mask])
    print(score)
from ycimpute.imputer import knnimput,iterforest
if __name__=="__main__":
    from datasets import load_data
    iris_miss,iris_ori=load_data.load_iris()
    X_filled = MIDA().complete(iris_miss)
    # X_filled = GAIN().complete(iris_miss)
    #X_filled =iterforest.MissForest().complete(iris_miss)
    # complete_data_, _, _ = min_max_scale(iris_ori)
    # X_filled, _, _ = min_max_scale(X_filled)
    #
    # score = evaluate.RMSE(complete_data_,
    #                       X_filled)
    import pandas as pd
    from utils.handle_missingdata import gene_missingdata
    path = r'../public_data/1_Iris.xlsx'
    data = pd.read_excel(path, sheet_name="dataset")
    dt = np.array(data.values)
    data = dt.astype('float')
    origin_data = data[:, :-1]
    target = data[:, -1]
    miss_data = gene_missingdata(rate=0.1, data=origin_data)
    print("ss")