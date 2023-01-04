'''
practice
R2: 0.55 이상
'''

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


# 1. Data
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
)

# print(x)
print(x.shape)
# print(y)
print(y.shape)


# # 2. model
# model = Sequential()
# model.add(Dense(100, input_dim=8))
# model.add(Dense(300))
# model.add(Dense(500))
# model.add(Dense(300))
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))


# # 3. compile and train
# model.compile(loss='mae')
# model.fit(x_train, y_train, epochs=2000, batch_size=1000)


# # 4. evaluate and predict
# loss = model.evaluate(x_test, y_test)
# print("Loss: ", loss)

# y_predict = model.predict(x_test)
# print("============")
# print(y_test)
# print(y_predict)
# print("============")

# from sklearn.metrics import mean_squared_error, r2_score
# def RMSE (y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# # def 함수_이름(para1, para2):
#     # return np.sqrt(mse)
# print("RMSE: ", RMSE(y_test, y_predict))

# r2 = r2_score(y_test, y_predict)
# print("R2: ", r2)

# '''
# print(dataset.feature_names)
# # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

# print(dataset.DESCR)
# '''