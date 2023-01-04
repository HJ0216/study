import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


# 1. Data
x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    shuffle=True,
    random_state=123
)


# 2. Model Construction
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


# 3. compile and train
model.compile(loss='mse', optimizer='adam',
              metrics=['mae', 'mse', 'accuracy', 'acc']) # accuracy = acc
# []: list, [] 안에는 2개 이상의 요소 대입 가능
# metrics 안에는 loss에 1개의 지표밖에 사용하지 못하므로 추가적으로 loss 타입을 사용하고 싶을 때 사용
model.fit(x_train, y_train, epochs=200, batch_size=1)


# 4. Evalueate and Predict
loss = model.evaluate(x_test, y_test)
# 최종 training을 기준으로 값이 나오는 게 아니므로 evalueate가 fit의 결과보다 안좋음
print("Loss: ", loss)



'''
Result

model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])
Epoch 200/200
14/14 [==============================] - 0s 2ms/step - loss: 9.5809 - mae: 2.0824 - mse: 9.5809
1/1 [==============================] - 0s 118ms/step - loss: 14.9538 - mae: 3.0450 - mse: 14.9538
Loss:  [14.953765869140625, 3.045029401779175, 14.953765869140625]

'''
