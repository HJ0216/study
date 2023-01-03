import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Data
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array(range(10)) # 0~9
# train set과 test set을 구분

# feature가 동일하므로 train data와 test data의 데이터 개수가 달라도 문제되지 않음
x_train = np.array([1, 2, 3, 4, 5, 6, 7]) # (7, )
x_test = np.array([8, 9, 10]) # (3, )

y_train = np.array(range(7)) # 0~6
y_test = np.array(range(7, 10)) # 7~9


# 2. Model Construction
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
# node shpae recommendation: 역 피라미드 or 다이아몬드


# 3. Compile and train
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1)


# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("Loss: ", loss)

result = model.predict([11])
print("Result: ", result)



'''
Result

model.fit(x_train, y_train, epochs=1000, batch_size=1)
Epoch 1000/1000
7/7 [==============================] - 0s 833us/step - loss: 0.1056

loss = model.evaluate(x_test, y_test)
1/1 [==============================] - 0s 114ms/step - loss: 0.3096

result = model.predict([11])
Result:  [[10.397283]]

'''