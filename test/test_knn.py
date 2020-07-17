#@Time:2019/12/13 17:15
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:test_knn.py
__author__ = "skux liu"

import pytest
from sklearn import datasets
from utils import kdtree

def test_knn_kdtree():
    iris = datasets.load_iris()
    data = iris.data
    data2=[tuple(i) for i in data]
    label = [i for i in range(len(data2))]
    tree=kdtree.create(data2)
    for i in range(len(data2)):
        print(data2[i])
        print(tree.search_knn(data2[i],1,distanceType='jaccard'))
    # kdtree2 = KNNKdTree(n_neighbors=1)
    # kdtree2.fit(data, label)
    # for i in range(len(data)):
    #     _, predict_label = kdtree2.predict(data[i])
    #     assert i==predict_label
if __name__=="__main__":
    # pytest.main()
    test_knn_kdtree()