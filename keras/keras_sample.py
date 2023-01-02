import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Data
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]]) # input_dim = 2
# np.array([[], []]): 다중 배열을 만들 경우, 그 안의 element의 개수는 일치 시켜야 함
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]) # output_dim = 1

# Data.shape: (행-데이터의 개수, 열-데이터의 특성, feature, column)
print(x.shape) # (2, 10)
print(y.shape) # (10, ) 1은 생략?

# 2. Model
model = Sequential()
model.add(Dense(5, input_dim=2)) # input_dim = 2(행렬에서 열의 개수), output = 5
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

# 3. Compile
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

# 4. Evaluate and Predict
loss = model.evaluate(x, y)
print("loss: ", loss)

result = model.predict([[10, 1.4]])
# predict(x1, x2)에서 x2 값을 알 수 없으므로 우선 훈련값을 넣어서 y와 유사한지 학인
print("[10, 1.4] result: ", result)


'''
Result
loss:
predict:
layer
'''