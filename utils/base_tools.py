"""
@author: liuAmon
@contact:utopfish@163.com
@file: base_tools.py
@time: 2020/7/15 23:44
"""
import os
import numpy as np
"""基础的未能单独分离出的函数"""

def merge_two_dicts(x, y):# type: (dict, dict) -> dict
    '''
    合并dict，相同key相加
    :param x:
    :param y:
    :return:
    '''
    z = dict()
    for key in x.keys():
        if key in y.keys():
            # 合并同类项
            x_value = x[key]
            y_value = y[key]
            if isinstance(x_value, dict) and isinstance(y_value, dict):
                result_x_y = merge_two_dicts(x_value, y_value)
            else:
                result_x_y = x_value + y_value
            z[key] = result_x_y
        else:
            z[key] = x[key]

    for key in y.keys():
        if key in x.keys():
            # 合并同类项
            x_value = x[key]
            y_value = y[key]
            if isinstance(x_value, dict) and isinstance(y_value, dict):
                result_x_y = merge_two_dicts(x_value, y_value)
            else:
                result_x_y = x_value + y_value
            z[key] = result_x_y
        else:
            z[key] = y[key]

    return z
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
def get_miss_location(missData):
    miss_location=[]
    for i in missData:
        temp = []
        for index, j in enumerate(i):
            if np.isnan(j):
                temp.append(index)
        miss_location.append(temp)
    return miss_location
def modifier (data,unqiue):
    data=data.copy()
    for i in range(len(data)):
        for j in range(len(data[0])):
                t = np.argmin(np.abs(unqiue-data[i][j]))
                data[i][j] = unqiue[t]
    return data
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