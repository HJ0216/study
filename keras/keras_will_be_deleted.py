import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# test_list = [[1, 2, 5, 8], 
#              [2, 3, 6, 9], 
#              [3, 4, 7, 10],
#              [4, 5, 8, 11]]

# test_array = np.array(test_list, int)

# print(test_array[1:2, :2]) # [[2 3]]
# # 1행 이상 2행 미만, 2열 미만

x = np.array([[[1,2,3],[4,5,6],[7,8,9]],
              [[1,2,3],[4,5,6],[7,8,9]]])

x2 = np.array([[[1,2,3],[4,5,6]],
               [[1,2,3],[4,5,6]],
               [[1,2,3],[4,5,6]],
               [[1,2,3],[4,5,6]]])

f = np.array([[[1, 2, 3],[4, 5, 6]],
              [[11, 22, 33],[44, 55, 66]],
              [[111, 222, 333],[444, 555, 666]]])

b = np.array([[1], [2], [3]])
print(b.shape) # (3, 1)

h = np.array([[[[1], [2], [3]],[[4], [5], [6]]],
              [[[11], [22], [33]],[[44], [55], [66]]]])
print(h.shape) # (2, 4, 1)





