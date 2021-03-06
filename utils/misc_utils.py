#@Time:2019/12/11 18:51
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:misc_utils.py
import numbers
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import multivariate_normal

def distance(point1,point2,type="euclidean"):
    newPoint1=[]
    newPoint2=[]
    for i,j in zip(point1,point2):
        if  not np.isnan(i) and not np.isnan(j):
            newPoint1.append(i)
            newPoint2.append(j)

    if type=="euclidean":
       return np.sqrt(np.sum(np.square(np.array(newPoint1) - np.array(newPoint2))))
    elif type=="jaccard":
       return pdist(np.vstack([newPoint1, newPoint2]), 'jaccard')[0]
    elif type=="hamming":
       return pdist(np.vstack([newPoint1, newPoint2]), 'hamming')[0]
    else:
       raise Exception("距离度量方法选择错误,请选择['euclidean','jaccard','hamming']")

def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def sortLabel(label):
    label = np.array(label)
    labelOld = []
    labelNum = len(list(set(label)))
    for i in label:
        if i not in labelOld:
            labelOld.append(i)
        if len(labelOld) == labelNum:
            break

    labelNew = sorted(labelOld)
    for i, old in enumerate(labelOld):
        label[label == old] = labelNew[i] + 10000
    return label - 10000

def prob(x, mu, cov):
    norm = multivariate_normal(mean=mu, cov=cov)
    return norm.pdf(x)

def log_prob(x, mu, cov):
    norm = multivariate_normal(mean=mu, cov=cov)
    return norm.logpdf(x)


def log_weight_prob(x, alpha, mu, cov):
    N = x.shape[0]
    return np.mat(np.log(alpha) + log_prob(x, mu, cov)).reshape([N, 1])

def RMSE(y_true, y_pred):
    '''
    Mean squared error
    '''
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def MAE(y_true, y_pred):
    '''
    Mean absolute error
    '''
    return np.mean(np.abs(y_true - y_pred))
def masked_mape_np(y_true, y_pred, null_val=0):
    '''
    MAPE
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100
def TF(y_true, y_pred):
    "相等差为0，不等为1"
    return np.mean(np.where(np.abs(y_true - y_pred)>0,1,0))
if __name__=="__main__":
    li1=np.array([1,2,3,4])
    li2=np.array([1,3,4,5])
    # print(np.square(li1-li2))
    # print(distance(li1,li2,"euclidean"))
    # print(distance(li1,li2,"jaccard"))
    # print(distance(li1,li2,"hamming"))
    print(RMSE(li1,li2))
    print(TF(li1,li2))