import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.datasets import load_boston

# 1. Data
# datasets = load_wine()

# plt.gray()
# plt.matshow(datasets.images[100])
# plt.show()


      
import numpy as np
'''
arr = np.array([[1 ,2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
arr = np.delete(arr, 0, axis=1)
print(arr)
[[ 2  3  4]
 [ 6  7  8]
 [10 11 12]]

'''

arr = np.array([[1 ,2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
arr = np.delete(arr, 0, axis=0)
'''
'''

