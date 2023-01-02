import numpy as np
import tensorflow as tf

# tensorflow version print
print(tf.__version__) # 2.7.4

# 1. 정제된 Data (정제된 데이터를 위해 전처리 필요)
x = np.array([1, 2, 3, 4, 5])
print(x) # [1 2 3 4 5]
y = np.array([1, 2, 3, 5, 4])

# 2. Model Construction (y=wx+b 구축)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))

# 3. Compile and training
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=200)

# 4. Evaluation and Prediction
results = model.predict([6])
print('predict 6: ', results)
