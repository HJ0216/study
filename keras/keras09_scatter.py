from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
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
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1)


# 4. Evalueate and Predict
loss = model.evaluate(x_test, y_test)
# 최종 training을 기준으로 값이 나오는 게 아니므로 evalueate가 fit의 결과보다 안좋음
print("Loss: ", loss)

y_predict = model.predict(x)
# predict 대입 값의 훈련 값 전체 대입
# fit의 훈련 결과를 predict에 사용
print("Result: ", y_predict)

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, y_predict, color="red")
plt.show()



'''
Result

Epoch 200/200
14/14 [==============================] - 0s 617us/step - loss: 1.9601
1/1 [==============================] - 0s 101ms/step - loss: 3.0334
Loss:  3.033447265625
Result:  [[ 1.0241305]
 [ 2.037987 ]
 [ 3.051844 ]
 [ 4.0657005]
 [ 5.079557 ]
 [ 6.0934143]
 [ 7.1072702]
 [ 8.121127 ]
 [ 9.134985 ]
 [10.14884  ]
 [11.162696 ]
 [12.176556 ]
 [13.19041  ]
 [14.204266 ]
 [15.218123 ]
 [16.23198  ]
 [17.245836 ]
 [18.259693 ]
 [19.27355  ]
 [20.287403 ]]
 
'''