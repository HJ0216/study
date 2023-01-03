import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Data
x = np.array([[range(10), range(21, 31), range(201, 211)],
              [range(10), range(21, 31), range(201, 211)]])

y = np.array([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]],
             [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]]]) # (2, 2, 10)

print(x.shape) # (2, 3, 10)
print(y.shape) # (2, 2, 10)

x = x.T
y = y.T

print(x.shape) # (2, 3, 10) input_dim=10
print(y.shape) # (2, 2, 10) output_dim=10


# 2. Model
# model = Sequential()
# model.add(Dense(10, input_dim=10)) # input_dim = 3
# model.add(Dense(10)) # output_dim = 2
# input_dim은 2차원까지만 사용 가능