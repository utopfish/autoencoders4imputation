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
    dataPath = r'C:\Users\pro\Desktop\实验相关文档\缺失插补建树全流程\04缺失插补结果'
    # analysisMeanMain(dataPath, dataPath)
    analysisPlot(dataPath, dataPath)