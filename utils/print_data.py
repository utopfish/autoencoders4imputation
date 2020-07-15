#@Time:2019/12/31 11:47
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:print_data.py
__author__ = "liuAmon"
'''
展示树长分布情况
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def find_martrix_min_value(data_matrix):
    new_data=[]
    for i in range(len(data_matrix)):
        new_data.append(min(data_matrix[i]))
    return min(new_data)

def statistics(data_matrix,min_data):
    hashmap={}
    for row in data_matrix-min_data:
        for val in row:
            if hashmap.get(val) is not None:
                hashmap[val]+=1
            else:
                hashmap[val]=1
    temp=sorted(hashmap.items(),key=lambda  x:x[0])
    val=[]
    count=[]
    for i in temp:
        val.append(int(i[0]))
        count.append(i[1])
    return val,count
def readRFfileAndPrint(path):
    data=pd.read_table(path,header=None)
    data=data.values
    newData=[[] for i in range(len(data))]
    for ind,i in enumerate(data):
            temp=i
            for j in range(len(temp)):
                try:
                    newData[ind].append(float(temp[j]))
                except:
                    continue
    temp=0
    for i in range(len(data)):

        temp+=np.sum(data[i][~np.isnan(data[i])])

    print(temp)
if __name__=="__main__":
    # path=r'C:\Users\pro\Desktop\tnt测试数据'
    # savePath=r'C:\Users\pro\Desktop\tnt测试数据plot'
    # for file in os.listdir(path):
    #     if file.endswith('out'):
    #         data=pd.read_table(os.path.join(path,file),header=None,sep=' ')
    #         data=data.values[:-1,1:].astype(float)
    #         min_data=find_martrix_min_value(data)
    #         val,count=statistics(data,min_data)
    #         #sns.barplot(x=val,y=count).set_title('liebherr')
    #         plt.bar(val,count)
    #         plt.title('{}'.format(file[:-4]))
    #         plt.savefig(os.path.join(savePath,'{}.jpg'.format(file[:-4])))
    #         plt.show()
    path=r'C:\Users\pro\Desktop\result_Aria2015.txt'
    readRFfileAndPrint(path)
    path2=r'C:\Users\pro\Desktop\result_Aria2015_fixed.txt'
    readRFfileAndPrint(path2)