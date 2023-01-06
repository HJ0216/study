import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1. Data
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=1234
)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


'''
x_train = x[:10]
y_train = y[:10]
x_test = x[10:13]
y_test = y[10:13]
x_validation = x[13:]
y_validation = y[13:]
# slicing [초과:이하]

print(x_train)
print(y_train)
print(x_test)
print(y_test)
print(x_validation)
print(y_validation)

'''


'''
# For train
x_train = np.array(range(1, 11)) # 1,2,3,4,5,6,7,8,9,10
y_train = np.array(range(1, 11))

# For evaluate
x_test = np.array([11,12,13])
y_test = np.array([11,12,13])

# For validatoin
x_validation = np.array([14,15,16])
y_validation = np.array([14,15,16])

# '''


# 2. Model
model = Sequential()
model.add(Dense(32, input_dim=1))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


# 3. compile and train
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=8,
          validation_split=0.25)
# validation_data를 통해서 val_loss 추가
# 훈련(train)보다 검증(validation)결과를 기준으로 테스트 결과를 판단해야 함
# validation_split을 통해서 x_train과 y_train 중 0.25의 validation 값 지정



# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

result = model.predict([17])
print("predict [17]: ", result)
