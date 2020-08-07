import matplotlib.pyplot as plt
import numpy as np
import torch
def as_num(x):
     y='{:.5f}'.format(x) # 5f表示保留5位小数点的float型
     return(y)


def merge_two_dicts(x, y):
    # type: (dict, dict) -> dict
    z = dict()
    for key in x.keys():
        if key in y.keys():
            # 合并同类项
            x_value = x[key]
            y_value = y[key]
            if isinstance(x_value, dict) and isinstance(y_value, dict):
                result_x_y = merge_two_dicts(x_value, y_value)
            else:
                result_x_y = x_value + y_value
            z[key] = result_x_y
        else:
            z[key] = x[key]

    for key in y.keys():
        if key in x.keys():
            # 合并同类项
            x_value = x[key]
            y_value = y[key]
            if isinstance(x_value, dict) and isinstance(y_value, dict):
                result_x_y = merge_two_dicts(x_value, y_value)
            else:
                result_x_y = x_value + y_value
            z[key] = result_x_y
        else:
            z[key] = y[key]

    return z
if __name__=="__main__":
    dict1 = {'a': 10, 'b': 8}
    dict2 = {'d': 6, 'c': 4,'a': 10}
    dict1=merge_two_dicts(dict1,dict2)
    print(dict1)
