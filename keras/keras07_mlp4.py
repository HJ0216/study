import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Data
x = np.array([range(10)]) # (10, )-최종 Vector (10, 1)-최종 Metirx 동일하게 취급
y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
             [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]) # (3, 10)

x = x.T # (10, 1), input_dim = 1
y = y.T # (10, 3), output_dim = 2

print(x.shape)
print(y.shape)


# 2. Model
model = Sequential()
model.add(Dense(10, input_dim=1)) # input_dim = 1
model.add(Dense(50))
model.add(Dense(90))
model.add(Dense(130))
model.add(Dense(170))
model.add(Dense(210))
model.add(Dense(170))
model.add(Dense(130))
model.add(Dense(90))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3)) # output_dim = 3


# 3. Compile
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=4)


# 4. (Evaluate) and Predict
result = model.predict([[10]]) # predict(x값, input_dim에 맞추기)
print("[10] result: ", result)



'''
# Result1

model.fit(x, y, epochs=1000, batch_size=2)
Epoch 1000/1000
5/5 [==============================] - 0s 998us/step - loss: 0.1198

result = model.predict([[10]])
[10] result:  [[10.696075   1.6338055 -0.7048712]]


# Result2: batch_size가 작아질 수록 loss가 올라감

model.fit(x, y, epochs=1000, batch_size=4)
3/3 [==============================] - 0s 1ms/step - loss: 0.1321

result = model.predict([[10]])
[10] result:  [[11.616149   1.7001956 -1.1217551]]

'''