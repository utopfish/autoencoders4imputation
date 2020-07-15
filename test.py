
from sklearn import preprocessing
import numpy as np
# ohc = preprocessing.OneHotEncoder()
#
# ohc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
#
# out=ohc.transform([[0, 1, 3]])
#
# recovered_X = ohc.inverse_transform(out)
# print(recovered_X)
print(preprocessing.scale([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]))