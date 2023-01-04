'''
[Practice]
trains set: 0.7 이상
평가지표1 R2: 0.8 이상
평가지표2 RMSE 사용
loss: mae or mse
optimizer: adam
matrics: optional
model(train set, epochs, batch_size=optional)
evaluate: x_test, y_test
r2 rmse: x_test -> y_predict && y_test
 
'''



import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split



# 1. Data
dataset = load_boston()
x = dataset.data # home
y = dataset.target # home_price

# print(x) # (506, 13)
# print(y) # (506, )


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
)



'''
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

print(dataset.DESCR)
'''



# 2. model
model = Sequential()
model.add(Dense(20, input_dim=13))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# 3. compile and train
model.compile(loss='mse')
model.fit(x_train, y_train, epochs=100, batch_size=2)

# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("Loss: ", loss)

y_predict = model.predict(x_test)

print("============")
print(y_test)
print(y_predict)
print("============")

from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
# def 함수_이름(para1, para2):
    # return np.sqrt(mse)
print("RMSE: ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2: ", r2)



'''
Result

RMSE:  7.6770552333648165
R2:  0.1517302885176257
'''