#@Time:2019/12/18 16:28
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:autoEncoder_pytorch.py
__author__ = "skux liu"
import os
import torch
from utils.random_impute import *
from auto_encoder import *
from utils.read_file import readNex
from utils.handle_missingdata import *
device=torch.device(bi['core'])


if __name__=="__main__":

    path=r'G:\labWork\cladistic-data-master\nexus_files'
    fileSavePath = r'G:\labWork\imputed_experiment_data\fix'
    # file='AhyongOM04crabs'

    modelSavePath = r'G:\labWork\imputed_experiment_data\model'
    for file in os.listdir(path):
    # file='AhyongOM04crabs.nex'
        try:
            data, misss_row, speciesname, begin, end = readNex(os.path.join(path,file))
            data=data+1
        except ValueError:
            print("可能存在数据多态问题")
            shear_dile(os.path.join(path,file),os.path.join("G:\labWork\cladistic-data-master\可能无用数据"))
            print("文件移动成功")
            continue
        try:
            imputed_data = random_inpute(data)
            saveData(fileSavePath, '{}_init.tnt'.format(file), speciesname, imputed_data, begin, end)

            logger.info("**********************{}********************".format(file))
            for  i in [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                result=np.zeros((5,12))
                for e in range(5):
                    # logger.info(" missing rate is {}".format(i))
                    # logger.info("epoch is {}".format(e))
                    miss_data= gene_missingdata(rate=i, data=imputed_data)

                    # logger.info("{}============".format(i))
                    # logger.info("random MSE:{}".format(MSE(imputed_data,random_inpute(miss_data))))

                    mask=get_mask(miss_data)
                    miss_location=get_miss_location(miss_data)
                    #数据均值化
                    # min_max_scaler = preprocessing.MinMaxScaler()
                    # mm_miss_data = min_max_scaler.fit_transform(miss_data)
                    min_max_scaler = preprocessing.StandardScaler()
                    mm_miss_data = min_max_scaler.fit_transform(miss_data)
                    modelName = file + str(i)
                    inp = interpolation_mask(modelName=modelName, completeData=random_inpute(mm_miss_data))
                    if not os.path.exists(os.path.join(modelSavePath, '{}.pkl'.format(modelName))):
                        inp.fit(os.path.join(modelSavePath, '{}.pkl'.format(modelName)),mask)

                    pred_data, fix_data = inp.prd(mm_miss_data, model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))

                    s = pd.DataFrame(np.unique(miss_data)).dropna(axis=0).values
                    s=[i[0] for i in s]

                    fix_auto_data = restore(min_max_scaler=min_max_scaler,s=s,data=fix_data)

                    pred_data=restore(min_max_scaler=min_max_scaler,s=s,data=pred_data)
                    # saveData(fileSavePath, '{}_{}_auto.tnt'.format(file, str(i / 2)), speciesname, fix_auto_data, begin, end)
                    # logger.info("{}============".format(i))
                    # logger.info("number of charater :{}".format(len(s)))
                    # logger.info("初步插补 MSE:{}".format(MSE(imputed_data, pred_data)))
                    # logger.info("ae MSE:{}".format(MSE(imputed_data,fix_auto_data)))
                    # logger.info("res change MSE:{}".format(MSE(pred_data, fix_auto_data)))

                    result[e,0]=(MSE(imputed_data, pred_data))
                    result[e,1]=(MSE(imputed_data,fix_auto_data))
                    result[e,2]=(MSE(pred_data, fix_auto_data))
                    # logger.info("初步插补 TF:{}".format(TF(imputed_data, pred_data)))
                    # logger.info("ae TF:{}".format(TF(imputed_data, fix_auto_data)))

                    X_filled_knn = KNN(k=3).fit_transform(mm_miss_data)
                    re_X = inp.revise(X_filled_knn, miss_location,
                                                 model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
                    X_filled_knn = restore(min_max_scaler=min_max_scaler,s=s,data=X_filled_knn)
                    re_X = restore(min_max_scaler=min_max_scaler, s=s, data=re_X)
                    # logger.info("knn MSE:{}".format(MSE(imputed_data, X_filled_knn)))
                    # logger.info("knn res MSE:{}".format(MSE(imputed_data, re_X)))
                    # logger.info("res change MSE:{}".format(MSE(X_filled_knn, re_X)))
                    result[e,3]=(MSE(imputed_data, X_filled_knn))
                    result[e,4]=(MSE(imputed_data, re_X))
                    result[e,5]=(MSE(X_filled_knn, re_X))
                    # X_filled_ii = IterativeImputer().fit_transform(mm_miss_data)
                    # re_X = inp.revise(X_filled_ii, miss_location,
                    #                   model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
                    # X_filled_ii = restore(min_max_scaler=min_max_scaler,s=s,data=X_filled_ii)
                    # re_X = restore(min_max_scaler=min_max_scaler, s=s, data=re_X)
                    # logger.info("ii MSE:{}".format(MSE(imputed_data, X_filled_ii)))
                    # logger.info("ii res MSE:{}".format(MSE(imputed_data,  re_X)))

                    X_filled_sf = SimpleFill().fit_transform(mm_miss_data)
                    re_X  = inp.revise(X_filled_sf, miss_location,
                                                 model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
                    X_filled_sf = restore(min_max_scaler=min_max_scaler,s=s,data=X_filled_sf)
                    re_X = restore(min_max_scaler=min_max_scaler, s=s, data=re_X)
                    # logger.info("sf MSE:{}".format(MSE(imputed_data, X_filled_sf)))
                    # logger.info("sf res MSE:{}".format(MSE(imputed_data, re_X)))
                    # logger.info("res change MSE:{}".format(MSE(X_filled_sf, re_X)))
                    result[e,6]=(MSE(imputed_data, X_filled_sf))
                    result[e,7]=(MSE(imputed_data, re_X))
                    result[e,8]=(MSE(X_filled_sf, re_X))
                    X_filled_me = SimpleFill("median").fit_transform(miss_data)
                    re_X = inp.revise(X_filled_me, miss_location,
                                      model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
                    X_filled_me = restore(min_max_scaler=min_max_scaler, s=s, data=X_filled_me)
                    re_X = restore(min_max_scaler=min_max_scaler, s=s, data=re_X)
                    # logger.info("median MSE:{}".format(MSE(imputed_data, X_filled_me)))
                    # logger.info("median res MSE:{}".format(MSE(imputed_data,re_X)))
                    # logger.info("res change MSE:{}".format(MSE(X_filled_me, re_X)))
                    result[e,9]=(MSE(imputed_data, X_filled_me))
                    result[e,10]=(MSE(imputed_data,re_X))
                    result[e,11]=(MSE(X_filled_me, re_X))
                    # result[e]=temp
                # logger.info(result)
                logger.info("number of charater :{}".format(len(s)))
                logger.info(" missing rate is {}".format(i))
                logger.info((result[0]+result[1]+result[2]+result[3]+result[4])/5)
        except Exception as e:
            logger.error(e)
            shear_dile(os.path.join(path, file), os.path.join("G:\labWork\cladistic-data-master\可能无用数据"))

