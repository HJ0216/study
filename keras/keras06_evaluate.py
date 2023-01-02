import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 1. Data
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# 2. Model
model = Sequential()
model.add(Dense(30, input_dim=1))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

# 3. Complile, training
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=7)

# 4. Evaluation, Prediction
loss = model.evaluate(x,y) # evaluete: loss 값 반환
# evaluate도 batch와 같이 다룰 수 있음
print("loss: ", loss)
# 훈련하는 데이터와 평가하는 데이터는 달라야함
# 만일 훈련 데이터(x, y)를 대입할 경우, model fit의 마지막 training loss와 model evaluate의 loss가 동일
result = model.predict([6])
print("6의 결과: ", result)

# 문제: loss와 result 중 어떤 값을 중심으로 예측치의 좋고 나쁨을 평가할 수 있을까
# -> loss가 낮을수록 최적의 weight에 가까워지므로 predict가 아닌 loss로 판단