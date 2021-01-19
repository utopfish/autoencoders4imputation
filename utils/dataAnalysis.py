# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : dataAnalysis.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/21 20:47
"""
from xlwt import *
import xlrd
from xlutils.copy import copy
import os
from prettytable import PrettyTable
import json
import numpy as np
def readJson(fileName):
    with open("{}".format(fileName),'r') as f:
        jsonData=json.load(f)
    return jsonData
def analysis(result):
    missRate = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    x = PrettyTable(['Method', 'pattern', 'missRate', 'RMSE', 'MAE', "MAPE"])
    rank = [[] for _ in range(len(missRate))]
    methods = []
    for patternMethod in result.keys():
        pattern = patternMethod.split(":")[0]
        method = patternMethod.split(":")[1]
        methods.append(method)
        for index, row in enumerate(result[patternMethod]):
            rank[index].append(row)
            try:
                temp = ["{:.4f}".format(i) for i in row]
            except:
                temp = ['NaN' for _ in row]
            x.add_row([method] + [pattern] + ["{:.2f}".format(missRate[index])] + temp)
    print(x)
    rank = np.array(rank)
    for index, missRateData in enumerate(rank):
        try:
            pos = np.where(missRateData == np.min(missRateData, axis=0))
            tmp = [methods[miscIndex] for miscIndex in list(pos[0])]
            print("缺失率为{}，下RMSE，MAP，MAPE最小值对应方法".format(missRate[index]), tmp)
        except Exception as e:
            print(e)
            continue
    top_k = 7
    for index, missRateData in enumerate(rank):
        try:
            missRateData = np.array(missRateData)
            arr = missRateData[:, 0]
            top_k_idx = arr.argsort()[0:top_k]
            tmp = [methods[miscIndex] for miscIndex in top_k_idx]
            print("缺失率为{}下,RMSE前三对应方法".format(missRate[index]), tmp)
            arr = missRateData[:, 1]
            top_k_idx = arr.argsort()[0:top_k]
            tmp = [methods[miscIndex] for miscIndex in top_k_idx]
            print("缺失率为{}下,MAP前三对应方法".format(missRate[index]), tmp)
            arr = missRateData[:, 2]
            top_k_idx = arr.argsort()[0:top_k]
            tmp = [methods[miscIndex] for miscIndex in top_k_idx]
            print("缺失率为{}下,MAPE前三对应方法".format(missRate[index]), tmp)
        except Exception as e:
            print(e)
            continue

totalResult={}
def analysisMean(file,result):
    #计算均值
    if file not in totalResult.keys():
        totalResult[file]={}
    missRate = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    for patternMethod in result.keys():
        t=result[patternMethod]
        for index, row in enumerate(result[patternMethod]):
            try:
                if "{}:{:.2f}".format(patternMethod, missRate[index]) not in totalResult[file].keys():
                    totalResult[file]["{}:{:.2f}".format(patternMethod, missRate[index])]=[]
                try:
                    temp = ["{:.4f}".format(i) for i in row]
                except:
                    temp = ['NaN' for _ in row]

                totalResult[file]["{}:{:.2f}".format(patternMethod, missRate[index])].append(temp)
            except Exception as e:
                print(e)
def getMeanAndStd(data):
    data=data.astype(np.float64)
    std=np.std(data)
    mean=np.mean(data)
    return mean,std


def analysisMeanMain(dataPath,savePath):
    for missRate in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for i in os.listdir(dataPath):
            if i.endswith('json'):
                result=readJson(os.path.join(dataPath,i))
                analysisMean("_".join(i.split('_')[:4]),result)
        x = PrettyTable(['dataSet','Method', 'pattern', 'missRate', 'RMSE', 'MAE', "MAPE"])
        matrixData=[[] for _ in range(3)]
        methodName=[i.split(":")[1] for i in result.keys()]
        for i in range(3):
            matrixData[i].append(['{}'.format(missRate)]+methodName)
        for datasetResult in totalResult:
            matrixDataTmp=[[datasetResult.split("_")[3]] for _ in range(3)]
            for missPatternMethodRate in totalResult[datasetResult]:
                missDataImputedMisc = np.array(totalResult[datasetResult][missPatternMethodRate])
                if float(missPatternMethodRate.split(":")[2])==missRate:
                    if  'MSELoss' not in missPatternMethodRate.split(":")[1] or \
                            'mice_MSELoss_Autoencoder' == missPatternMethodRate.split(":")[1] :
                        tmp = []
                        for i in range(3):
                            M, S = getMeanAndStd(missDataImputedMisc[:, i])
                            tmp.append("{:.3f}+{:.3f}".format(M, S))
                            matrixDataTmp[i].append("{:.3f}+{:.3f}".format(M, S))
                        x.add_row([datasetResult]+missPatternMethodRate.split(":")  + tmp)
            for i in range(3):
                matrixData[i].append(matrixDataTmp[i])
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        for i,loss in enumerate(['MSE','MAP','MAPE']):
            write_excel_xls(os.path.join(savePath,"{}.xls".format(loss)),"data",matrixData[i])
        print(x)


def json2Excel(filePath,ExcelPath):
    """
    将Json文件转为Excel文件
    :param filePath:
    :param ExcelPath:
    :return:
    """
    for i in os.listdir(filePath):
        if i.endswith('json'):
            result = readJson(os.path.join(filePath, i))
            print(result)
def write_excel_xls(path, sheet_name, value):
    index = len(value)  # 获取需要写入数据的行数
    if os.path.exists(path):
        workbook = xlrd.open_workbook(path, formatting_info=True)  # 打开工作簿
        rows = workbook.sheets()[0].nrows+1
        workbook = copy(workbook)
        sheet = workbook.get_sheet(0)
    else:
        workbook = Workbook()  # 新建一个工作簿
        sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
        rows=0

    style = XFStyle()
    pattern = Pattern()
    pattern.pattern = Pattern.SOLID_PATTERN
    pattern.pattern_fore_colour = Style.colour_map['yellow']  # 设置单元格背景色为黄色
    style.pattern = pattern
    for i in range(rows, rows+index):
        try:
            tmp=[float(MeanStd.split("+")[0]) for MeanStd in value[i-rows][1:]]
            minIndex=tmp.index(min(tmp))+1

        except Exception as e:
            minIndex=-1
        for j in range(0, len(value[i-rows])):
            if j==minIndex:
                sheet.write(i, minIndex, value[i-rows][minIndex], style)
            else:
                sheet.write(i, j, value[i-rows][j])  # 像表格中写入数据（对应的行和列）

    workbook.save(path)
def analysisPlot(dataPath,savePath):
    for missRate in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for i in os.listdir(dataPath):
            if i.endswith('json'):
                result=readJson(os.path.join(dataPath,i))
                analysisMean("_".join(i.split('_')[:4]),result)
        x = PrettyTable(['dataSet','Method', 'pattern', 'missRate', 'RMSE', 'MAE', "MAPE"])
        matrixData=[[] for _ in range(3)]
        for i in range(3):
            matrixData[i].append(['{}'.format(missRate),'Random',	'median',	'EM','ours'])
        for datasetResult in totalResult:
            matrixDataTmp=[[datasetResult.split("_")[3]] for _ in range(3)]
            for missPatternMethodRate in totalResult[datasetResult]:
                missDataImputedMisc = np.array(totalResult[datasetResult][missPatternMethodRate])
                if float(missPatternMethodRate.split(":")[2])==missRate:
                        tmp = []
                        for i in range(3):
                            M, S = getMeanAndStd(missDataImputedMisc[:, i])
                            tmp.append("{:.3f}".format(M))
                            matrixDataTmp[i].append("{:.3f}".format(M))
                        x.add_row([datasetResult]+missPatternMethodRate.split(":")  + tmp)
            for i in range(3):
                matrixData[i].append(matrixDataTmp[i])
        for i,loss in enumerate(['MSE_plt','MAP_plt','MAPE_plt']):
            write_excel_xls(os.path.join(savePath,"{}.xls".format(loss)),"data",matrixData[i])
        print(x)
if __name__=="__main__":

    dataPath=r'E:\labCode\autoencoders4imputation\experiment\publicData'
    savePath=r'C:\Users\pro\Desktop\自编码器部分实验\原始全数据全方法'
    analysisMeanMain(dataPath,savePath)


