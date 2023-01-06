# kaggle.com/competitions/bike-sharing-demand
# casual, registered, count 삭제
# datetime -> idx 처리


import numpy as np
import pandas as pd
import time


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# 1. Data
path = './_data/bike/'
train_csv = pd.read_csv(path+'train.csv', index_col=0)
print(train_csv) # [10886 rows x 11 columns]

test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission = pd.read_csv(path+'sampleSubmission.csv', index_col=0)

print(train_csv.isnull().sum())
# data_set의 결측치(Null) 값 총계 출력
train_csv = train_csv.dropna()
# pandas.dropna(): null 값을 포함한 데이터 행 삭제
print(train_csv.shape) # (10886, 11)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
# column 명이 casual, registered 'count'(axis=1)인 column 삭제
# evaluate에 필요한 test_cvs와 동일한 형태를 

print(x)
# [10886 rows x 8 columns] -> dropna로 인한 변경

y = train_csv['count']
print(y) # Length: 10886


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    shuffle=True,
    train_size=0.7,
    random_state=123
)

print(x_train.shape, x_test.shape) # (7620, 8), (3266, 8)
print(y_train.shape, y_test.shape) # (7620,), (3266,)


# 2. model
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16))
model.add(Dense(1)) # output_dim = 1
# Activation default: linear

# output_dim에서 activation을 'sigmoid'를 사용할 경우, return value가 0(predict값이 0.5미만일 경우) or 1(predict값이 0.5이상일 경우)로만 반환하게 됨
# -> 이진 분류에서만 sigmoid를 마지막에 사용
# Hidden Layer에서 sigmoid를 사용할 수 있으나, 값이 너무 한정적으로 변하기때문에 이진 분류를 제외한 곳에서 사용을 권장하지 않음
# relu도 hidden layer에서만 사용 권장, output_dim에서 사용 시, 음수값이 왜곡될 가능성이 있음

'''
Case1
activation = linear
RMSE


Case2
activation = sigmoid
RMSE


Case3
activation = relu
RMES


'''

# 3. compile and train
model.compile(loss='mse', optimizer='adam')
# RMSE가 평가지표이므로 유사한 mse 사용
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32)
end = time.time()


# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("Loss: ", loss)

y_predict = model.predict(x_test)
# test는 y값이 없으므로 train file의 x_test로 splited된 파일 사용
print(y_predict)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE: ", rmse)

print("소요 시간: ", end - start)

'''
(tf27) cpu 소요 시간:  18.1283917427063
(tf274gpu) gpu 소요 시간:  77.01734590530396
'''

# for submission
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape) # (6493, 1)

print(submission)
submission['count'] = y_submit
# pandas(submission['count'])에 numpy(y_submit)를 직접 대입시키면 numpy가 pandas가 됨
print(submission)

submission.to_csv(path+'sampleSubmission_1.csv')