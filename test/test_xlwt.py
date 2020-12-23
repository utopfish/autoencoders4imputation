# -*- coding: utf-8 -*-
"""
@Project : autoencoders4imputation
@File    : test_xlwt.py
@Author  : Mr.Liu Meng
@E-mail  : utopfish@163.com
@Time    : 2020/12/23 13:35
"""
from xlwt import *

def excel_write(data_list):
    book = Workbook(encoding='utf-8')
    sheet = book.add_sheet('Sheet1') #创建一个sheet

    #创建一个样式----------------------------
    style = XFStyle()
    pattern = Pattern()
    pattern.pattern = Pattern.SOLID_PATTERN
    pattern.pattern_fore_colour = Style.colour_map['yellow'] #设置单元格背景色为黄色
    style.pattern = pattern
    #-----------------------------------------

    sheet.write(0, 0, label = 'ICAO') #给第0行的第1列插入值
    sheet.write(0, 1, label = 'Location') #给第0行的第2列插入值
    sheet.write(0, 2, label='Airport_Name')
    sheet.write(0, 3, label='Country')

    #循环插入值
    for num,x in enumerate(data_list):
        index=num+1
        print(index)
        # if index!=0:
        sheet.write(index, 0, label = x["ICAO"])
        sheet.write(index, 1, label = x["Location"])
        sheet.write(index, 2, label=x["Airport_Name"], style=style) #将样式添加到此单元格
        sheet.write(index, 3, label=x["Country"])
    book.save('air.xls')
#测试数据
data=[{"ICAO": "DSG", "Location": "SDGSDG", "Airport_Name": "sdgsdg??sdg",
     "Country": "dfsdg"},{"ICAO": "DSG", "Location": "SDGS23G", "Airport_Name": "sdgsdg23??sdg",
     "Country": "354746"}]
excel_write(data)