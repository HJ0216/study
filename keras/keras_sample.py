import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. data
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])
c = np.array([[1, 2, 3]])
d = np.array([[1, 2],
              [3, 4],
              [5, 6]])
e = np.array([[1, 2, 3, 4],
             [11, 22, 33, 44],
             [111, 222, 333, 444]])
f = np.array([[[1, 2, 3],[4, 5, 6]],
              [[11, 22, 33],[44, 55, 66]],
              [[111, 222, 333],[444, 555, 666]]])
g = np.array([[[1, 2], [3, 4], [5, 6]],
             [[1, 2], [3, 4], [5, 6]]])
h = np.array([[[[1], [2], [3]],[[4], [5], [6]]],
              [[[11], [22], [33]],[[44], [55], [66]]]])
i = np.array([[[[1, 2], [3, 4], [5, 6], [7, 8]],
              [[11, 22], [33, 44], [55, 66], [77, 88]]],
              [[[1, 2], [3, 4], [5, 6], [7, 8]],
              [[11, 22], [33, 44], [55, 66], [77, 88]]]])
j = np.array([[[[1,2]]]])
# z = np.array([[1,2,3][4,5]]) # TypeError

print("a.shape: ", a.shape) # Vector: (3, )
print("b.shape: ", b.shape) # Metrix: (3, 1)
print("c.shape: ", c.shape) # Metirx: (1, 3)
print("d.shape: ", d.shape) # Metrix: (3, 2)
print("e.shape: ", e.shape) # Metrix: (3, 4)
print("f.shape: ", f.shape) 
print("g.shape: ", g.shape)
print("h.shape: ", h.shape)
print("i.shape: ", i.shape)
print("j.shape: ", j.shape)
# print("z.shape: ", z.shape)


model = Sequential()
model.add(Dense(10, input_dim=3)) # input_dim = 3