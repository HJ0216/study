import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

'''
* reshpae
CNN conv2D 처리하기 위해 4D화
(60000, 28, 28) -> (60000, 28, 28, 1)
'''

print(np.unique(y_train, return_counts=True))
'''
데이터 특성 파악
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(filters=128,
                 kernel_size=(2, 2),
                 input_shape=(28, 28, 1),
                 activation='relu')) # (27, 27, 128)
model.add(Conv2D(filters=64,
                 kernel_size=(2, 2))) # 26, 26, 64
model.add(Conv2D(filters=64,
                 kernel_size=(2, 2))) # 25, 25, 64
model.add(Flatten())
model.add(Dense(32, activation='relu'))  # 32- 임의
model.add(10, activation='softmax') # 10=y_class

model.summary()