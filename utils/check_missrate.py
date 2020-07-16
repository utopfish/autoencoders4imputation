#@Time:2020/1/19 15:46
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:check_missrate.py
__author__ = "liuAmon"
import os
import numpy as np
from utils.read_file import readNex

if __name__=="__main__":
    '''
    检查数据的缺失率
    '''
    path=r'C:\Users\pro\Desktop\all_nex_data'
    for file in os.listdir(path):
        try:
            if file.endswith('nex'):
                data, miss_row, speciesName = readNex(os.path.join(path, file))
                missCount=0
                for i in range(len(data)):
                    for j in range(len(data[i])):
                        if np.isnan(data[i][j]):
                            missCount+=1
                missRate=missCount/(len(data)*len(data[0]))
                print(file,missRate)
        except Exception as e :
            print(e)

