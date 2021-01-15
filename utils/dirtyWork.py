# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : dirtyWork.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2021/1/15 19:17
"""
import re
def bayes2Tre(path,savePath):
    res = []
    speciesName={}
    #是否开始获取种名的标志
    flag=0

    with open(path, "r") as f:  # 打开文件
        data = f.read()
        for i in data.split('\n')[1:-1]:
            if 'translate' in i:
                flag=1
                continue
            if flag==1 and "end;" not in i:

                if "[&U]" not in i :
                    tmp=i.replace(",","").split(" ")
                    speciesName[tmp[-2]]=tmp[-1]
                else:
                    i = re.sub(":[0123456789e\.-]+", "", i.split("[&U] ")[1])
                    for j in range(len(speciesName) - 1, 0, -1):
                        i = i.replace('{}'.format(j), '{}'.format(speciesName[str(j)]))
                    res.append(i)
            # # count(i)
            #
            # print(i)

    with open(savePath,'w') as f:
        for i in res[:200]:
            f.writelines(i+"\n")


if __name__=="__main__":
    path=r'C:\Users\pro\Desktop\bayes\MrBayes\0.5_03_Dikow2009.nex.run1.t'
    savePath=r'C:\Users\pro\Desktop\bayes\0.5_03_Dikow2009.nex.run1.txt'
    bayes2Tre(path,savePath)