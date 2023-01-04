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
# []: list, [] 안에는 2개 이상의 요소 대입 가능
# metrics 안에는 loss에 1개의 지표밖에 사용하지 못하므로 추가적으로 loss 타입을 사용하고 싶을 때 사용
model.fit(x_train, y_train, epochs=200, batch_size=1)


# 4. Evalueate and Predict
loss = model.evaluate(x_test, y_test)
# 최종 training을 기준으로 값이 나오는 게 아니므로 evalueate가 fit의 결과보다 안좋음
print("Loss: ", loss)

y_predict = model.predict(x_test)
# 비교해야할 대상, x_test에 대한 예측값의 predict -> y_test

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



from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
# def 함수_이름(para1, para2):
    # return np.sqrt(mse)

print("RMSE: ", RMSE(y_test, y_predict)) # root(MSE)




'''
Result

'''