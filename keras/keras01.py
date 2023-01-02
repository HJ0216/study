import tensorflow as tf
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
#mae: min absolute error, loss값을 최소로 하기위해 mae를 사용
#adam: loss 최적화 = adam 사용
model.fit(x, y, epochs=10)
#model.fit: training start
# epochs: training 횟수(횟수가 너무 많아지면, 오히려 틀어지는 경우가 생길 수 있음)

# execute: ctrl + F5
# Result
# Epoch 1/10 : 실행 횟수 현황
# 1/1 [==============================] - 0s 200ms/step - loss: 4.4002 : loss가 실행 시 마다 줄어감

model.fit(x, y, epochs=100)
# 초기 랜덤값이 다르므로 epochs 마다 loss가 달라짐

# 4. Valuation and Prediction
result = model.predict([4])
print('Result: ', result)
# print('문자 인식', 실제 출력 값)

model.fit(x, y, epochs=1000)
result = model.predict([4])
print('Result2: ', result)

model.fit(x, y, epochs=3000)
result = model.predict([4])
print('Result3: ', result)
