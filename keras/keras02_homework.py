import numpy as np
# numpy: 고성능의 다차원 배열 객체와 이를 다룰 도구를 제공하는 라이브러리

# Target: predict [13]

# 1. Refined Deta
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# numpy library의 array function: array(object(array))
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 2. Model Construction(y=ax+b를 구현하기 위해 import)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Dense implements the operation: output = activation(dot(input, kernel) + bias)
# where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer.
# Dense: output(y) = kernel(weights) * input(x) + bias

model = Sequential()
# Creates a Sequential model instance.
# Sequential: provides training and inference(추론) features on this model.

model.add(Dense(1, input_dim=1))
# dense(첫번째 인자 : 출력 뉴런의 수,  input_dim : 입력 뉴런의 수)

# 3. compile and training(best weight, minimum loss)
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=2000)

# 4. Evaluation and Prediction
result = model.predict([13])
print('Result: ', result)