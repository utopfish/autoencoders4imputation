#@Time:2019/12/31 16:47
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:readFile.py
__author__ = "liuAmon"
'''
读文件函数
'''
import pandas as pd
import numpy as np
def readNex(path):
    info=[]
    speciesname=[]
    with open(path, "r") as f:  # 打开文件
        flag = 0
        # data =
        split_data=f.read().split('\n')# 读取文件
        begin = []
        end = []
        for i in split_data:
            if 'MATRIX' in i or 'matrix' in i or 'Matrix' in i :
                flag = 1
                begin.append(i)
                continue
            elif ';' == i:
                flag = 2
                end.append(i)
                continue
            if flag == 0:
                begin.append(i)
            elif flag == 2:
                end.append(i)
            elif flag == 1 and i != '':
                i = i.replace('\t', ' ')
                speciesname.append(i.strip().split(' ')[0])
                info.append(''.join(i.strip().split(' ')[1:]).replace("\t", ""))
    data=[[] for i in range(len(info))]
    for i in range(len(info)):
        for j, val in enumerate(info[i]):
            try:
                data[i].append(int(val))
            except:
                if val == "-":
                    data[i].append(-1)
                elif val == "?":
                    data[i].append(np.nan)
    misss_row = []
    for ind, i in enumerate(data):
        if np.nan in i:
            misss_row.append(ind)
            continue
    return np.array(data,dtype=float),misss_row,speciesname,begin,end
import os
def saveData(fileSavePath,fileName,speciesName,data,begin,end):
    with  open(os.path.join(fileSavePath, fileName), 'w') as f:
        temp = list(data)
        for i in begin:
            f.writelines(i+'\n')
        for i in range(len(temp)):
            dataTemp=[]
            for j in range(len(temp[i])):
                try:
                    dataTemp.append(str(int(temp[i][j])))
                    if dataTemp[-1]=='-1':
                        dataTemp[-1]='-'
                except:
                    dataTemp.append('?')
            # temp[i] = ['-' if  str(int(x)) for x in temp[i]]
            temp[i] = speciesName[i] + ' ' + ''.join(dataTemp)
            f.writelines(temp[i] + '\n')
        for i in end:
            f.writelines(i+'\n')
def readTreeSpecies(path,speciesName):
    with open(path, "r") as f:  # 打开文件
        data = f.read()
        for i in data.split('\n'):
            for ind,j in enumerate(speciesName):
                i=i.replace(j,'{}'.format(ind))
            i=i.replace(',',' ')
            print(i)

def readDistFile(path):
    data=pd.read_table(path,sep='\t',header=None)
    data=data.values
    res=[]
    for i in data:
        res+=i[0].strip().split(" ")[1:]
    res=[float(i) for  i in res]
    return res
def Aria2015_rf():
    path = r'C:\Users\pro\Desktop\int_data\TreeInAll\Aria2015_tree0_rf.txt'
    data=readDistFile(path)
    origin=np.sum(data[2:102])/100
    auto=np.sum(data[102:202])/100
    ii=np.sum(data[202:302])/100
    knn=np.sum(data[302:402])/100
    me=np.sum(data[402:502])/100
    sf=np.sum(data[502:602])/100
    path = r'C:\Users\pro\Desktop\int_data\TreeInAll\Aria2015_tree1_rf.txt'
    data = readDistFile(path)
    origin1 = np.sum(data[2:102]) / 100
    auto1 = np.sum(data[102:202]) / 100
    ii1 = np.sum(data[202:302]) / 100
    knn1 = np.sum(data[302:402]) / 100
    me1 = np.sum(data[402:502]) / 100
    sf1 = np.sum(data[502:602]) / 100

    print(path.split('\\')[-1])
    print("origin",(origin+origin1)/2)
    print('auto',(auto+auto1)/2)
    print('ii',(ii+ii1)/2)
    print('knn',(knn+knn1)/2)
    print('me',(me+me1)/2)
    print('sf',(sf+sf1)/2)
def Aria2015_co():
    path = r'C:\Users\pro\Desktop\int_data\TreeInAll\Aria2015_tree1_co.txt'
    data=readDistFile(path)
    origin=np.sum(data[2:102])/100
    auto=np.sum(data[102:202])/100
    ii=np.sum(data[202:302])/100
    knn=np.sum(data[302:402])/100
    me=np.sum(data[402:502])/100
    sf=np.sum(data[502:602])/100
    path = r'C:\Users\pro\Desktop\int_data\TreeInAll\Aria2015_tree1_co.txt'
    data = readDistFile(path)
    origin1 = np.sum(data[2:102]) / 100
    auto1 = np.sum(data[102:202]) / 100
    ii1 = np.sum(data[202:302]) / 100
    knn1 = np.sum(data[302:402]) / 100
    me1 = np.sum(data[402:502]) / 100
    sf1 = np.sum(data[502:602]) / 100
    print(path.split('\\')[-1])
    print("origin",(origin+origin1)/2)
    print('auto',(auto+auto1)/2)
    print('ii',(ii+ii1)/2)
    print('knn',(knn+knn1)/2)
    print('me',(me+me1)/2)
    print('sf',(sf+sf1)/2)
def Longrich2010():
    path = r'C:\Users\pro\Desktop\int_data\TreeInAll\Longrich2010_co.txt'
    data=readDistFile(path)
    origin=np.sum(data[1:51])/50
    auto=np.sum(data[51:101])/50
    ii=np.sum(data[101:151])/50
    knn=np.sum(data[151:201])/50
    me=np.sum(data[201:251])/50
    sf=np.sum(data[251:301])/50

    print(path.split('\\')[-1])
    print("origin",origin)
    print('auto',auto)
    print('ii',ii)
    print('knn',knn)
    print('me',me)
    print('sf',sf)

    path = r'C:\Users\pro\Desktop\int_data\TreeInAll\Longrich2010_rf.txt'
    data = readDistFile(path)
    origin = np.sum(data[1:51]) / 50
    auto = np.sum(data[51:101]) / 50
    ii = np.sum(data[101:151]) / 50
    knn = np.sum(data[151:201]) / 50
    me = np.sum(data[201:251]) / 50
    sf = np.sum(data[251:301]) / 50

    print(path.split('\\')[-1])
    print("origin", origin)
    print('auto', auto)
    print('ii', ii)
    print('knn', knn)
    print('me', me)
    print('sf', sf)

def Dikow2009():
    path = r'C:\Users\pro\Desktop\int_data\TreeInAll\Dikow2009_co.txt'
    data=readDistFile(path)
    origin=np.sum(data[1:101])/100
    auto=np.sum(data[101:201])/100
    ii=np.sum(data[201:301])/100
    knn=np.sum(data[301:401])/100
    me=np.sum(data[401:501])/100
    sf=np.sum(data[501:601])/100

    print(path.split('\\')[-1])
    print("origin",origin)
    print('auto',auto)
    print('ii',ii)
    print('knn',knn)
    print('me',me)
    print('sf',sf)

    path = r'C:\Users\pro\Desktop\int_data\TreeInAll\Dikow2009_rf.txt'
    data = readDistFile(path)
    origin = np.sum(data[1:101]) / 100
    auto = np.sum(data[101:201]) / 100
    ii = np.sum(data[201:301]) / 100
    knn = np.sum(data[301:401]) / 100
    me = np.sum(data[401:501]) / 100
    sf = np.sum(data[501:601]) / 100

    print(path.split('\\')[-1])
    print("origin", origin)
    print('auto', auto)
    print('ii', ii)
    print('knn', knn)
    print('me', me)
    print('sf', sf)
def Liu2011():
    path = r'C:\Users\pro\Desktop\int_data\TreeInAll\Liu2011_co.txt'
    data=readDistFile(path)
    origin=np.sum(data[1:101])/100
    ii=np.sum(data[101:201])/100
    knn=np.sum(data[201:301])/100
    me=np.sum(data[301:401])/100
    sf=np.sum(data[401:501])/100


    print(path.split('\\')[-1])
    print("origin",origin)
    print('ii',ii)
    print('knn',knn)
    print('me',me)
    print('sf',sf)

    path = r'C:\Users\pro\Desktop\int_data\TreeInAll\Liu2011_rf.txt'
    data = readDistFile(path)
    data = readDistFile(path)
    origin = np.sum(data[1:101]) / 100
    ii = np.sum(data[101:201]) / 100
    knn = np.sum(data[201:301]) / 100
    me = np.sum(data[301:401]) / 100
    sf = np.sum(data[401:501]) / 100

    print(path.split('\\')[-1])
    print("origin", origin)
    print('ii', ii)
    print('knn', knn)
    print('me', me)
    print('sf', sf)
if __name__=="__main__":
    # Aria2015_rf()
    # Aria2015_co()
    Longrich2010()
    Dikow2009()
    Liu2011()
    # path=r'C:\Users\pro\Desktop\all_nex_data\Aria2015.nex'
    # data,miss_row,speciesName=readNex(path)
    # print(data)
    # print(miss_row)
    # print(speciesName)
    # for ind,i in enumerate(speciesName):
    #     print(ind,i)
    # treeSpecies=r'C:\Users\pro\Desktop\Aria2015paper_result.txt'
    # readTreeSpecies(treeSpecies,speciesName)