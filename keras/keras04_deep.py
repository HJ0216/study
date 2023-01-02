import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense # for Model
from tensorflow.keras.models import Sequential # for Model

# 1. Data
x = np.array([1,2,3,4,5]) # List 1개
y = np.array([1,2,3,5,4])

# 2. Model
model = Sequential() # Sequential class
model.add(Dense(30, input_dim=1)) # Dense: Layer, (Output, input)
model.add(Dense(50)) # Layer 쌓기
model.add(Dense(70)) # 이전 layer의 output이 다음 layer의 input이 되므로 생략 가능
model.add(Dense(90))
model.add(Dense(90))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))
# 생각해볼 것: layer와 node의 모양(다이아몬드, 피라미드 등)의 구성을 어떻게 해야 효율적인 ML이 될까

# 3. Complile, training
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=500)

# 4. Evaluation, Prediction
result = model.predict([6])
print('6의 결과: ', result)

# Hyper-parameter tuning: loss를 줄이기 위해 epochs, node, layer, batch_size 등 변경
# epochs 높이기
# node의 개수 늘리기(model.add(Dense(1)))
# layer의 계층 늘리기
# batch_size 조절하기