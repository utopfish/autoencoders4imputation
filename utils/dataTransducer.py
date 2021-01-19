# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : dataTransducer.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2021/1/18 17:12
"""
## 主要进行数据格式转换
import pandas as pd

import re
def bayes2Tre(path,savePath):
    # """
    # 将贝叶斯结果转化为treespace可读的树
    # :param path:Mr bayes输出文件.run1.t 路径
    # :param savePath:保存文件路径
    # :return:
    # sample：
    # path=r'C:\Users\pro\Desktop\bayes\0.5_03_Dikow2009.nex.run1.t'
    # savePath=r'C:\Users\pro\Desktop\Aguado2009_200树\0.5_03_Dikow2009.bayes.txt'
    # bayes2Tre(path,savePath)
    # """

    res = []
    speciesName={}
    #是否开始获取种名的标志
    flag=0
    count=0
    with open(path, "r") as f:  # 打开文件
        data = f.read()
        for i in data.split('\n')[1:-1]:
            if 'translate' in i:
                flag=1
                continue
            if flag==1 and "end;" not in i:

                if "[&U]" not in i :
                    tmp=i.replace(",","").replace(";","").split(" ")
                    speciesName[tmp[-2]]=tmp[-1]
                else:
                    i = re.sub(":[0123456789e\.\-\+]+", "", i.split("[&U] ")[1])
                    for j in range(len(speciesName), 0, -1):
                        i = i.replace('{}'.format(j), '{}'.format(speciesName[str(j)]))
                    res.append(i)
                    count+=1
                    print(count,i)
    with open(savePath,'w') as f:
        for i in res[:200]:
            f.writelines(i+"\n")


#将R模拟生成数据转化为nex数据
def sim2nex(path,savePath):
    data=pd.read_csv(path,index_col=0)
    value=data.values
    #注意修改分类单元数和特征数
    begin="#NEXUS \n begin data; \ndimensions ntax = 203 nchar = 200; \nformat datatype = standard gap = - missing =? Interleave = no SYMBOLS = \"1234\";\nmatrix \n"
    end=";\nEND;\n"
    with open(savePath,'w') as f:
        f.write(begin)
        for index,val in enumerate(value):
            t=[str(i) for i in val]
            f.writelines("sp{} ".format(index+1)+"".join(t)+"\n")
        f.write(end)

if __name__=="__main__":
    path=r'C:\Users\pro\Desktop\ConstructTreeProject\sim.csv'
    savePath=r'C:\Users\pro\Desktop\ConstructTreeProject\sim.nex'
    sim2nex(path,savePath)