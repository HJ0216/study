import tensorflow as tf
# tf version print
print(tf.__version__)

import numpy as np

# 주석 표시

# 1. 정제된 Deta
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])


# 2. Model Construction(y=ax+b를 구현하기 위해 import)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense #tensorflow.keras.layers의 Dense import

model = Sequential()
model.add(Dense(1, input_dim=1))
# Dense dimension을 add
# add(Dense(y배열([1, 2, 3])-output, x배열([1, 2, 3])-input))
# Dense: y = wx + b 의 함수를 한 번


# 3. compile and training(best weight, minimum loss)
model.compile(loss='mae', optimizer='adam')
# mae: min absolute error, loss값을 최소로 하기위해 mae를 사용
# adam: loss 최적화 = adam 사용
model.fit(x, y, epochs=10)
# model.fit: training start
# epochs: training 횟수(횟수가 너무 많아지면, 오히려 틀어지는 경우가 생길 수 있음)

model.fit(x, y, epochs=100)
# 초기 랜덤값이 다르므로 실행 시 마다 loss가 달라짐


# 4. Evaluation and Prediction
result = model.predict([4])
print('Result1: ', result)
# print('문자 인식', 실제 출력 값)

# 5. ect
model.fit(x, y, epochs=1000)
result = model.predict([4])
# result 값이 덮어씌어짐
print('Result2: ', result)

model.fit(x, y, epochs=3000)
result = model.predict([4])
print('Result3: ', result)



'''
Result1

Epoch 100/100
1/1 [==============================] - 0s 2ms/step - loss: 8.6560e-04
Result1:  [[4.001488]]


Result2

Epoch 1000/1000
1/1 [==============================] - 0s 999us/step - loss: 0.0035
Result2:  [[3.9903853]]


Result3

Epoch 3000/3000
1/1 [==============================] - 0s 1ms/step - loss: 2.3774e-04
Result3:  [[4.000531]]


'''