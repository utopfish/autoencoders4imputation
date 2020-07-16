#@Time:2019/12/24 16:45
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:handler_loss.py
__author__ = "liuAmon"
import numpy as np
def MSE(array1,array2):

    return np.sum(np.square(array1-array2))

def TF(array1,array2):
    t = np.square(array1 - array2).flatten()
    result=[1 if i!=0 else 0 for i in t]
    return np.sum(result)
if __name__=="__main__":
    test1=np.ones((3,3))
    test2=np.zeros((3,3))
    print(MSE(test2,test1))