# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : publicDataAnalysis.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/23 15:20
"""
from utils.dataAnalysis import analysisMeanMain
if __name__=="__main__":
    dataPath = '../experiment\publicData'
    analysisMeanMain(dataPath, dataPath)