"""
@author: liuAmon
@contact:utopfish@163.com
@file: base_impute.py
@time: 2020/7/16 0:01
"""
import numpy as np
import pandas as pd
import random
def random_inpute(data):
    '''
        :param data: 含有缺失数据集
        :return:对不可适用和缺失进行随机插补
        '''
    new_data=data.copy()
    for row in range(len(new_data)):
        for col in range(len(new_data[0])):
            if np.isnan(new_data[row][col]) or new_data[row][col]==-1:
                t=pd.DataFrame(new_data[:,col]).dropna(axis=0).values
                t=set([i[0]  for i in t])
                t.discard(-1)
                new_data[row][col]=random.choice(list(t))
    return np.array(new_data,dtype=float)