# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : test_tqdm.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/22 20:35
"""
import tqdm
pbar = tqdm.tqdm(["a", "b", "c", "d"], desc='dirs')
for char in pbar:
    pbar.set_description("Processing %s" % char)