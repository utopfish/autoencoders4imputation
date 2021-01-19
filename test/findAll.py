# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : findAll.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2021/1/19 20:47
"""
import re
s=[i.start() for i in re.finditer('s', 'ssssss')]
print(s)