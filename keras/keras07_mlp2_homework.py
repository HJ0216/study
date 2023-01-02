import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1. data
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
             [9, 8, 7, 6, 5, 4, 3 ,2, 1, 0]])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

print(x.shape)
print(y.shape)

x = x.T

print(x.shape)
# x = np.array([1,1,9], 
#              [2,1,8], 
#              [3,1,7], 
#              [4,1,6], 
#              [5,2,5], 
#              [6,1.3,4], 
#              [7,1.4,3], 
#              [8,1.5,2], 
#              [9, 1.6,1], 
#              [10, 1.4,0])


# 2. Model
model = Sequential()
model.add(Dense(10, input_dim=3)) # input_dim = 3(입력값 기준으로 열의 개수, Input Layer), output = 10(hidden layer로 임의값 설정 가능)
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1)) # output_dim = 1(출력값 기준으로 열의 개수, Output Layer)


# 3. Compile
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=300, batch_size=2)
# 2행씩 훈련 5번(5/5)


# 4. Evaluate and Predict
loss = model.evaluate(x, y)
print("loss: ", loss)

result = model.predict([[10, 1.4, 0]])
print("[10, 1.4, 0] result: ", result)



'''
Result

model.fit(x, y, epochs=300, batch_size=2)
Epoch 300/300
5/5 [==============================] - 0s 750us/step - loss: 0.1140

loss = model.evaluate(x, y)
1/1 [==============================] - 0s 91ms/step - loss: 0.0352

result = model.predict([[10, 1.4, 0]])
[10, 1.4, 0] result:  [[20.014843]]

'''