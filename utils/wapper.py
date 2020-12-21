# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : wapper.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/21 22:39
"""

import time
def costTime(func):
    def wapper(*args, **kwargs):
        start=time.time()
        tmp=func(*args, **kwargs)
        print("{}：耗时{}".format(func.__name__,str(time.time()-start)))
        return tmp
    return wapper