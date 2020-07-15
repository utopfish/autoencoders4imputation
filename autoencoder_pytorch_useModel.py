#@Time:2019/12/23 14:15
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:autoencoder_pytorch_useModel.py
__author__ = "liuAmon"
import torch
import pandas as pd
from sklearn import preprocessing
from utils.handlerMissData import geneMissData
from autoEncoder_pytorch import AE,Encoder,Decoder

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(4, 2)
decoder = Decoder(2, 4)
ae = AE(encoder, decoder)
ae.load_state_dict(torch.load('iris.pkl'))
iris = pd.read_excel('public_data/1_Iris.xlsx', sheet_name="dataset")
dt = iris.values
iris = dt.astype('double')
# 归一化，去掉标签
min_max_scaler = preprocessing.MinMaxScaler()
iris = min_max_scaler.fit_transform(iris[31:50, :-1])
iris_miss=geneMissData(0.2,iris)

criterion=torch.nn.MSELoss()
inputs=torch.ones(1,4,requires_grad=True,device=device)
print(inputs.is_leaf)
for ind,i in enumerate([0.5556, 0.1250, 0.5763, 0.5000]):
   inputs.data[0,ind]=i
learning_rate=1
for t in range(100):
    outputs=decoder(encoder(inputs))

    loss=criterion(outputs,inputs)
    loss.backward()
    if t%20==0:
        print("input:{}".format(inputs))
        print("inputs grad：{}".format(inputs.grad))
        print("loss:{}".format(loss))
    with torch.no_grad():
        inputs.data[0,2]-=learning_rate* inputs.grad.data[0,2]
        #inputs -= learning_rate * inputs.grad
        inputs.grad.zero_()




