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

x = np.array([[[1],[2]]])
print(x.shape)

