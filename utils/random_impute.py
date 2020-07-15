import numpy as np
import pandas as pd
import random
from utils.readFile import readNex,saveData
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

if __name__=="__main__":
    path=r'G:\labWork\cladistic-data-master\nexus_files\AhyongOM04crabs.nex'
    fileSavePath=r'G:\labWork\imputed_experiment_data'
    data,misss_row,speciesname,begin,end=readNex(path)
    imputed_data=random_inpute(data)
    saveData(fileSavePath,'{}_init.tnt'.format('AhyongOM04crabs'), speciesname, imputed_data, begin, end)
    print(imputed_data)