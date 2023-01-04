# practice
# 1. <<R2를 음수가 아닌 0.5 이하로 줄이기>>
# 2. Data: 유지
# 3. Layer: 인풋 및 아웃풋 포함 7개 이상
# 4. batch_size = 1
# 5. hidden layer node: 각각 10~100
# 6. train set: 70%
# 7. epochs>=100
# 8. loss: mse or mae
# 9. activation 사용 금지

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split


# 1. Data
x = np.array(range(1,21))
y = np.array(range(1,21))

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    shuffle=True,
)


# 2. Model Construction
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))


# 3. compile and train
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)


# 4. Evalueate and Predict
loss = model.evaluate(x_test, y_test)
print("Loss: ", loss)

y_predict = model.predict(x_test)

print("============")
print(y_test)
print(y_predict)
print("============")



'''
Result
============
[ 9  7  5 23  8  3]
[[14.496417]
 [ 6.170927]
 [ 5.245874]
 [17.271582]
 [ 8.94609 ]
 [ 8.021036]]
 ============
'''



from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
# def 함수_이름(para1, para2):
    # return np.sqrt(mse)
print("RMSE: ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R: ", r2)



'''
Result

RMSE:  3.8482795786702315
-> loss이므로 낮을수록 성능이 좋음
R:  0.6485608399723322
-> accuracy이므로 높을수록 성능이 좋음
'''