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
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=200, batch_size=1)


# 4. Evalueate and Predict
loss = model.evaluate(x_test, y_test)
print("Loss: ", loss)

y_predict = model.predict(x_test)
print("==========")
print(y_test)
print(y_predict)
print("==========")
# y_predict: parameter로 x_test를 대입하여 나온 y_predict값과 y_test 값을 비교



'''
Result

==========
[ 9  7  5 23  8  3]
[[14.428579 ]
 [ 6.1343384]
 [ 5.2127547]
 [17.193329 ]
 [ 8.899085 ]
 [ 7.9775033]]
==========
'''



from sklearn.metrics import mean_squared_error

def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
# def 함수_이름(para1, para2):
    # return np.sqrt(mse), root(MSE)

print("RMSE: ", RMSE(y_test, y_predict))

'''
RMSE:  3.8499891113446925
RMSE:  3.858623138575582
RMSE:  3.857018758234503
'''



'''
Result

(28) model.compile(loss='rmse', optimizer='adam', metrics=['mae'])
(31) model.fit(x_train, y_train, epochs=200, batch_size=1)

Error Location: File "c:\study\keras\keras12_RMSE.py", line 31
-> 오류가 발생한 위치의 근처를 안내
ValueError: Unknown loss function -> loss type에는 rmse가 없음

'''
