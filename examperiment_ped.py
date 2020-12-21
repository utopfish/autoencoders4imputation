#@Time:2019/12/30 20:59
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:examperiment_ped.py
__author__ = "liuAmon"

import os
import numpy as np
from sklearn import preprocessing
from utils.handle_missingdata import gene_missingdata
from utils.read_file import readNex,saveData
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler,SimpleFill
from logger import logger
from iris_buldModel import get_miss_location,interpolation
def modifier (data,unqiue):
    data=data.copy()
    for i in range(len(data)):
        for j in range(len(data[0])):
            # t=np.where(np.abs(unqiue-data[i][j])==np.min(np.abs(unqiue-data[i][j])))

                t = np.argmin(np.abs(unqiue-data[i][j]))
                data[i][j] = unqiue[t]


    return data
from torch.utils import data
if __name__=="__main__":
    # for file in os.listdir(r'C:\Users\pro\Desktop\all_nex_data'):
    # filePath=r'C:\Users\ASUS\Desktop\usefulDataSimple\usefulData'
    # modelSavePath=r'C:\Users\ASUS\Desktop\usefulDataSimple\Model'
    # fileSavePath=r'C:\Users\ASUS\Desktop\usefulDataSimple\fixed'
    filePath=r'G:\labWork\imputed_experiment_data'
    modelSavePath=r'G:\labWork\imputed_experiment_data\model'
    fileSavePath=r'G:\labWork\imputed_experiment_data\fix'
    for file in os.listdir(filePath):
        if file.endswith('tnt'):
            file = file[:-4]
            for i in [0.1,0.2,0.4,0.5]:
                try:
                    # file='02Bennett94pterosaurs'

                # file='Liu2011'
                # originData,miss_mask,speciesName=readNex(r'C:\Users\pro\Desktop\all_nex_data\{}.nex'.format(file))
                    originData, miss_mask, speciesName,begin,end = readNex(os.path.join(filePath,'{}.tnt'.format(file)))
                    missData, miss_mask = gene_missingdata(rate=i, data=originData)



                    try:
                        min_max_scaler = preprocessing.MinMaxScaler()
                        data = min_max_scaler.fit_transform(missData)
                        miss_location = get_miss_location(data[miss_mask])
                        modelName = file+str(i)
                        inp = interpolation(modelName=modelName, completeData=np.delete(data, miss_mask, axis=0))
                        if not os.path.exists(os.path.join(modelSavePath,'{}.pkl'.format(modelName))):
                            inp.fit(os.path.join(modelSavePath,'{}.pkl'.format(modelName)))
                        pred_data, fix_data = inp.prd(data[miss_mask],model=os.path.join(modelSavePath,'{}.pkl'.format(modelName)))
                        s=np.unique(np.delete(missData,miss_mask,axis=0))
                        inv_auto_data=min_max_scaler.inverse_transform(fix_data)
                        inv_auto_data=modifier(inv_auto_data,s)
                        fix_auto_data=missData.copy()
                        fix_auto_data[miss_mask]=inv_auto_data
                        fix_auto_data=fix_auto_data.astype(int)
                        saveData(fileSavePath,'{}_{}_auto.tnt'.format(file,str(i/2)),speciesName,fix_auto_data,begin,end)
                        saveData(fileSavePath,'{}_{}_missdata.tnt'.format(file, str(i / 2)), speciesName, missData, begin, end)
                        try:
                            X_filled_knn = KNN(k=3).fit_transform(missData)
                            X_filled_knn = X_filled_knn.astype(int)
                            saveData(fileSavePath,'{}_{}_knn.tnt'.format(file, str(i / 2)), speciesName, X_filled_knn, begin, end)
                        except Exception as e:
                            print(e)
                        try:
                            X_filled_ii = IterativeImputer().fit_transform(missData)
                            X_filled_ii = X_filled_ii.astype(int)
                            saveData(fileSavePath,'{}_{}_ii.tnt'.format(file, str(i / 2)), speciesName, X_filled_ii, begin, end)
                        except Exception as e:
                            print(e)
                        try:
                            X_filled_sf = SimpleFill().fit_transform(missData)
                            X_filled_sf = X_filled_sf.astype(int)
                            saveData(fileSavePath,'{}_{}_sf.tnt'.format(file, str(i / 2)), speciesName, X_filled_sf, begin, end)
                        except Exception as e:
                            print(e)
                        try:
                            X_filled_me = SimpleFill("median").fit_transform(missData)
                            X_filled_me = X_filled_me.astype(int)
                            saveData(fileSavePath,'{}_{}_me.tnt'.format(file, str(i / 2)), speciesName, X_filled_me, begin, end)
                        except Exception as e:
                            print(e)
                    except Exception as e :
                        logger.info(e)
                # with  open(os.path.join(fileSavePath,'{}_auto.txt'.format(file)), 'w') as f:
                #     temp=list(originData)
                #     for i in range(len(temp)):
                #         temp[i]=['-' if x==-1 else str(x) for x in temp[i]]
                #         temp[i]=speciesName[i]+' '+''.join(temp[i])
                #         f.writelines(temp[i]+'\n')


                except Exception as e:
                    logger.info(file+":"+str(e))
