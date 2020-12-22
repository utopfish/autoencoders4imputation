#@Time:2019/12/31 16:47
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:read_file.py
__author__ = "liuAmon"
'''
读文件函数
'''
import pandas as pd
import numpy as np
import xlrd
import xlwt
from xlutils.copy import copy
import os
import impyute
from logger import logger
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
def read_result_log(path,save_path):
    '''
    读取方法实验结果,读取pub结果
    :param path str:
    :return:
    '''
    with open(path) as file_obj:
        content = file_obj.read()
        resu=[]
        temp={}

        for i in content.split("\n"):
            if "271 -" in i :
                temp['name']=i.split("271 - file:")[1].split("\n")[0].replace("*","")
            if "272 -" in i:
                temp['pattern']=i.split("272 - pattern :")[1].split("\n")[0]
            if "273 -" in i:
                temp['rate']=i.split("273 - ")[1].split("\n")[0]
            if "277 -" in i:
                temp['mice']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "279 -" in i:
                temp['median']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "281 -" in i:
                temp['random']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "283 -" in i:
                temp['mida']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "285 -" in i:
                temp['gain']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "287 -" in i:
                temp['tai_mice']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "290 -" in i:
                temp['tresai_mice']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "292 -" in i:
                temp['tresai_ii']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]

            if "297 -" in i:
                temp['mice_all']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "299 -" in i:
                temp['ii_all']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "301 -" in i:
                temp['median_all']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "303 -" in i:
                temp['random_all']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "305 -" in i:
                temp['tai_mice_all']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "308 -" in i:
                temp['tresai_mice_all']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "310 -" in i:
                temp['tresai_ii_all']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
                resu.append(temp)
                temp={}

    import csv
    f = open(save_path, 'w',newline='' ,encoding='utf-8')
    csv_writer = csv.writer(f)
    for row in range(len(resu)):
        csv_writer.writerow(["file:"+resu[row]['name']])
        csv_writer.writerow(['Method','pattern','RMSE','MAE',"MAPE"])
        for i in ['mice','median','random','mida','gain','tai_mice','tresai_mice','tresai_ii','',
                  'mice_all','median_all','random_all','tai_mice_all','tresai_mice_all','tresai_ii_all']:
            if i=="":
                csv_writer.writerow(['--','--','-','--','--'])
            else:
                try:
                    temp=["%.4f" % float(a) for a in resu[row][i]]
                except:
                    temp=['NaN' for _ in resu[row][i]]

                csv_writer.writerow([i] +[resu[row]['pattern']]+temp)
    f.close()
def read_result_log2(path,save_path):
    '''
    读取方法实验结果,读取ped结果
    :param path str:
    :return:
    '''
    with open(path) as file_obj:
        content = file_obj.read()
        resu=[]
        temp={}

        for i in content.split("\n"):
            if "285 - file:" in i :
                temp['name']=i.split("285 - file:")[1].split("\n")[0].replace("*","")
            if "286 -" in i:
                temp['pattern']=i.split("286 - pattern :")[1].split("\n")[0]
            if "287 - " in i:
                temp['rate']=i.split("287 - ")[1].split("\n")[0]
            if "289 - " in i:
                temp['mice']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "291 -" in i:
                temp['ii']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "293 -" in i:
                temp['median']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "295 -" in i:
                temp['random']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "297 -" in i:
                temp['mida']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "299 -" in i:
                temp['gain']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "301 -" in i:
                temp['tai_mice']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "304 -" in i:
                temp['tresai_mice']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "306 -" in i:
                temp['tresai_ii']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]

            if "311 -" in i:
                temp['mice_all']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "313 -" in i:
                temp['ii_all']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "315 -" in i:
                temp['median_all']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "317 -" in i:
                temp['random_all']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "319 -" in i:
                temp['tai_mice_all']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "322 -" in i:
                temp['tresai_mice_all']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
            if "324 -" in i:
                temp['tresai_ii_all']=["".join(filter(lambda ch: ch in '0123456789.',j)) for j in i.split(" - ")[1].split("\n")[0].split(',') ]
                resu.append(temp)
                temp={}

    import csv
    f = open(save_path, 'w',newline='' ,encoding='utf-8')
    csv_writer = csv.writer(f)
    for row in range(len(resu)):
        csv_writer.writerow(["file:"+resu[row]['name']])
        csv_writer.writerow(['Method','pattern','RMSE','MAE',"MAPE"])
        for i in ['mice','median','random','mida','gain','tai_mice','tresai_mice','tresai_ii','',
                  'mice_all','median_all','random_all','tai_mice_all','tresai_mice_all','tresai_ii_all']:
            if i=="":
                csv_writer.writerow(['--','--','-','--','--'])
            else:
                try:
                    temp=["%.4f" % float(a) for a in resu[row][i]]
                except:
                    temp=['NaN' for _ in resu[row][i]]

                csv_writer.writerow([i] +[resu[row]['pattern']]+temp)
    f.close()


def write_excel_xls(path, sheet_name, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
    workbook.save(path)
def write_excel_xls_append(path, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
    new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i + rows_old, j, value[i][j])  # 追加写入数据，注意是从i+rows_old行开始写入
    new_workbook.save(path)  # 保存工作簿

    #     csv_writer.writerow(['Method', 'pattern', 'RMSE', 'MAE', "MAPE"])
    #     #     for i in ['mice', 'median', 'random', 'mida', 'gain', 'tai_mice', 'tresai_mice', 'tresai_ii', '',
    #     #               'mice_all', 'median_all', 'random_all', 'tai_mice_all', 'tresai_mice_all', 'tresai_ii_all']:
    #     #         if i == "":
    #     #             csv_writer.writerow(['--', '--', '-', '--', '--'])
    #     #         else:
    #     #             try:
    #     #                 temp = ["%.4f" % float(a) for a in resu[row][i]]
    #     #             except:
    #     #                 temp = ['NaN' for _ in resu[row][i]]
    #     #
    #     #             csv_writer.writerow([i] + [resu[row]['pattern']] + temp)
    #     # f.close()
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

def readAllTypeFile(file):
    if file.endswith('.xlsx'):
        try:
            data = pd.read_excel(file, sheet_name="dataset")
            dt = np.array(data.values)
            data = dt.astype('float')
            originData = data[:-1]
            target = data[-1]
            return originData
        except Exception as e:
            print(e)
            logger.error("文件读取错误：{}".format(file))
            return None
    elif file.endswith('.nex'):
        try:
            data, misss_row, speciesname, begin, end = readNex(file)
            # 之后进行测试是否需要在原数基础上+1 或+10
            data = data
            originData = impyute.imputation.cs.random(data)
            return originData
        except ValueError:
            print("可能存在数据多态问题")
            # shear_dile(os.path.join(path, file), os.path.join("G:\labWork\cladistic-data-master\可能无用数据"))
            print("文件移动成功")
            return None
    elif file.endswith('.data'):
        try:
            dataset = pd.read_csv(file)
            data = np.array(dataset.T)
            originData = data[:-1]
            target = data[-1]
            return originData
        except Exception as e:
            print(e)
            return None
    elif file.endswith('.csv'):
        data = pd.read_csv(file)
        dt = np.array(data.values)
        data = dt.astype('float')
        originData = data[:-1]
        target = data[-1]
        return originData
if __name__=="__main__":
    # Aria2015_rf()
    # Aria2015_co()
    # Longrich2010()
    # Dikow2009()
    # Liu2011()
    # path=r'E:\labCode\autoencoders4imputation\logs\2020-08-07-13h.log'
    # read_result_log2(path,'tai_tresai_test02.csv')
    path='../public_data'
    for file in os.listdir(path):
        print(file)
        data=readAllTypeFile(os.path.join(path,file))
        # if data==None:
        #     print(file)
        # print(data)
