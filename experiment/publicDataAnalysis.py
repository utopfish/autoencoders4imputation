# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : publicDataAnalysis.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/23 15:20
"""
from utils.dataAnalysis import analysisMeanMain,analysisPlot
if __name__=="__main__":
    # dataPath = r'C:\Users\pro\Desktop\autoencoders4imputation\experiment\加大自编码器参数'
    # savePath = r'C:\Users\pro\Desktop\实验二自编码器建树\UCI数据集测试\加大自编码器参数'
    # analysisMeanMain(dataPath, savePath)


    # dataPath = r'C:\Users\pro\Desktop\autoencoders4imputation\experiment\publicData'
    # savePath = r'C:\Users\pro\Desktop\实验二自编码器建树\UCI数据集测试\原始全数据全方法'
    # analysisMeanMain(dataPath, savePath)


    dataPath = r'C:\Users\pro\Desktop\autoencoders4imputation\experiment\加大自编码器参数-mice'
    savePath = r'C:\Users\pro\Desktop\实验二自编码器建树\UCI数据集测试\加大参数后自编码器与MICE单独比较'
    analysisMeanMain(dataPath, savePath)
