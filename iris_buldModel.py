#@Time:2019/12/25 17:47
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:iris_buldModel.py
__author__ = "liuAmon"
import time
from utils import kdtree
from utils.random_impute import *
from sklearn import preprocessing
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, IterativeImputer, BiScaler,SimpleFill
from utils.handlerLoss import MSE

class interpolation():
    def __init__(self,feature_length=None,h_length=None,modelName=None,completeData=None):
        self.batch_size=50
        self.epochs=20
        self.learning_rate=0.01
        self.iterations=20
        self.feature_length=feature_length
        self.h_length=h_length
        self.modelName=modelName
        self.completeData = completeData
    def fit(self,path):
        if self.feature_length==None:
            self.feature_length=len(self.completeData[0])
            self.h_length=self.feature_length//2
        dataloader = torch.utils.data.DataLoader(self.completeData, batch_size=self.batch_size,
                                                 shuffle=True, num_workers=2)
        encoder = Encoder(self.feature_length, self.h_length)
        decoder = Decoder(self.h_length, self.feature_length)
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
                loss = criterion(dec, inputs)
                loss.backward()
                optimizer.step()
                l = loss.item()
            print(epoch, l)
            if np.isnan(l):
                break
            if l < bestLoss:
                bestLoss = l
                torch.save(ae.state_dict(), path)
                logger.info("{}'s loss is {}".format(path,l))
                print("模型保存成功")
    def prd(self,missdata,model):
        try:
            if self.feature_length==None:
                self.feature_length=len(self.completeData[0])
                self.h_length=self.feature_length//2
            km = KMeansBase(3, distanceType="euclidean")
            labels = km.fit_predict(self.completeData)
            data = [tuple(i) for i in km.centers]
            tree = kdtree.create(data)
            pred_data = []  # 预测数据
            miss_mark = []  # 记录缺失位置
            for i in missdata:
                temp = []
                for index, j in enumerate(i):
                    if np.isnan(j):
                        temp.append(index)
                miss_mark.append(temp)
            for ind, i in enumerate(missdata):
                t = tree.search_knn(i, 1, distanceType='euclidean')
                closeNode = list(t[0][0].data)
                interped_dat = list(i.copy())
                for ind in range(len(interped_dat)):
                    if np.isnan(interped_dat[ind]):
                        interped_dat[ind] = closeNode[ind]
                pred_data.append(interped_dat)
            pred_data = np.array(pred_data)

            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device = torch.device(bi['core'])
            encoder = Encoder(self.feature_length, self.h_length)
            decoder = Decoder(self.h_length, self.feature_length)
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
                        for mark in miss_mark[num]:
                            inputs.data[0, mark] -= self.learning_rate * inputs.grad.data[0, mark]
                        inputs.grad.zero_()
                fixNum[num] = inputs.data.cpu().numpy()
            return pred_data,fixNum
        except Exception as e:
            logger.info(e)
    def revise(self,beforeRevisedata,miss_loaction):
        pred_data = np.array(beforeRevisedata)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(bi['core'])
        encoder = Encoder(self.feature_length, self.h_length)
        decoder = Decoder(self.h_length, self.feature_length)
        ae = AE(encoder, decoder)
        ae.load_state_dict(torch.load('Model/{}.pkl'.format(self.modelName), map_location=bi['core']))
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
from auto_encoder import *
if __name__=="__main__":
    start=time.time()

    # #归一化，去掉标签
    path = r'public_data/'
    fileSavePath = r'G:\labWork\imputed_experiment_data\fix'
    # file='AhyongOM04crabs'

    modelSavePath = r'G:\labWork\imputed_experiment_data\model'
    for file in os.listdir(path):
        logger.info("**********************{}********************".format(file))
        data = pd.read_excel(os.path.join('public_data',file), sheet_name="dataset")
        dt = np.array(data.values)
        data = dt.astype('float')
        data=data[:-1]
        target=data[-1]
        for i in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            result = np.zeros((5, 12))
            for e in range(5):
                # logger.info(" missing rate is {}".format(i))
                # logger.info("epoch is {}".format(e))
                miss_data = geneMissData(rate=i, data=data)

                # logger.info("{}============".format(i))
                # logger.info("random MSE:{}".format(MSE(imputed_data,random_inpute(miss_data))))

                mask = get_mask(miss_data)
                miss_location = get_miss_location(miss_data)
                # 数据均值化
                # min_max_scaler = preprocessing.MinMaxScaler()
                # mm_miss_data = min_max_scaler.fit_transform(miss_data)
                min_max_scaler = preprocessing.StandardScaler()
                mm_miss_data = min_max_scaler.fit_transform(miss_data)
                modelName = file + str(i)
                inp = interpolation_mask(modelName=modelName, completeData=random_inpute(mm_miss_data))
                if not os.path.exists(os.path.join(modelSavePath, '{}.pkl'.format(modelName))):
                    inp.fit(os.path.join(modelSavePath, '{}.pkl'.format(modelName)), mask)

                pred_data, fix_data = inp.prd(mm_miss_data, model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))

                s = pd.DataFrame(np.unique(miss_data)).dropna(axis=0).values
                s = [i[0] for i in s]

                fix_auto_data = min_max_scaler.inverse_transform(fix_data)

                pred_data = min_max_scaler.inverse_transform(pred_data)
                # saveData(fileSavePath, '{}_{}_auto.tnt'.format(file, str(i / 2)), speciesname, fix_auto_data, begin, end)
                # logger.info("{}============".format(i))
                # logger.info("number of charater :{}".format(len(s)))
                # logger.info("初步插补 MSE:{}".format(MSE(imputed_data, pred_data)))
                # logger.info("ae MSE:{}".format(MSE(imputed_data,fix_auto_data)))
                # logger.info("res change MSE:{}".format(MSE(pred_data, fix_auto_data)))

                result[e, 0] = (MSE(data, pred_data))
                result[e, 1] = (MSE(data, fix_auto_data))
                result[e, 2] = (MSE(pred_data, fix_auto_data))
                # logger.info("初步插补 TF:{}".format(TF(imputed_data, pred_data)))
                # logger.info("ae TF:{}".format(TF(imputed_data, fix_auto_data)))

                X_filled_knn = KNN(k=3).fit_transform(mm_miss_data)
                re_X = inp.revise(X_filled_knn, miss_location,
                                  model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
                X_filled_knn = min_max_scaler.inverse_transform(X_filled_knn)
                re_X = min_max_scaler.inverse_transform(re_X)
                # logger.info("knn MSE:{}".format(MSE(imputed_data, X_filled_knn)))
                # logger.info("knn res MSE:{}".format(MSE(imputed_data, re_X)))
                # logger.info("res change MSE:{}".format(MSE(X_filled_knn, re_X)))
                result[e, 3] = (MSE(data, X_filled_knn))
                result[e, 4] = (MSE(data, re_X))
                result[e, 5] = (MSE(X_filled_knn, re_X))
                # X_filled_ii = IterativeImputer().fit_transform(mm_miss_data)
                # re_X = inp.revise(X_filled_ii, miss_location,
                #                   model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
                # X_filled_ii = restore(min_max_scaler=min_max_scaler,s=s,data=X_filled_ii)
                # re_X = restore(min_max_scaler=min_max_scaler, s=s, data=re_X)
                # logger.info("ii MSE:{}".format(MSE(imputed_data, X_filled_ii)))
                # logger.info("ii res MSE:{}".format(MSE(imputed_data,  re_X)))

                X_filled_sf = SimpleFill().fit_transform(mm_miss_data)
                re_X = inp.revise(X_filled_sf, miss_location,
                                  model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
                X_filled_sf = min_max_scaler.inverse_transform(X_filled_sf)
                re_X = min_max_scaler.inverse_transform(re_X)
                # logger.info("sf MSE:{}".format(MSE(imputed_data, X_filled_sf)))
                # logger.info("sf res MSE:{}".format(MSE(imputed_data, re_X)))
                # logger.info("res change MSE:{}".format(MSE(X_filled_sf, re_X)))
                result[e, 6] = (MSE(data, X_filled_sf))
                result[e, 7] = (MSE(data, re_X))
                result[e, 8] = (MSE(X_filled_sf, re_X))
                X_filled_me = SimpleFill("median").fit_transform(miss_data)
                re_X = inp.revise(X_filled_me, miss_location,
                                  model=os.path.join(modelSavePath, '{}.pkl'.format(modelName)))
                X_filled_me = min_max_scaler.inverse_transform(X_filled_me)
                re_X = min_max_scaler.inverse_transform(re_X)
                # logger.info("median MSE:{}".format(MSE(imputed_data, X_filled_me)))
                # logger.info("median res MSE:{}".format(MSE(imputed_data,re_X)))
                # logger.info("res change MSE:{}".format(MSE(X_filled_me, re_X)))
                result[e, 9] = (MSE(data, X_filled_me))
                result[e, 10] = (MSE(data, re_X))
                result[e, 11] = (MSE(X_filled_me, re_X))
                # result[e]=temp
            # logger.info(result)
            logger.info("number of charater :{}".format(len(s)))
            logger.info(" missing rate is {}".format(i))
            logger.info((result[0] + result[1] + result[2] + result[3] + result[4]) / 5)
