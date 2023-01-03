import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


# 1. Data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array(range(10)) # [0,1,2,3,4,5,6,7,8,9]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.7,
    shuffle = True,
    random_state=1)
print("x_train, x_test: ", x_train, x_test, "\ny_train, y_test: ", y_train, y_test)


'''
train_test_split(arrays, test_size, train_size, random_state, shuffle, stratify)

parameter
arrays: 분할시킬 Data
test_size: 전체 Data 중 test로 사용할 test set 비율
train_size: 1 - test_size (생략 가능)
random_state: 입력 시, 함수 수행 시 마다 결과가 바뀌지 않음
* 같은 데이터로 계속 훈련을 해줘야하므로 random_state를 입력해줘야 함
shuffle: 셔플 여부(Default = True)
stratify: 해당 Data 비율 유지
Ex. data = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
0의 비율: 70%, 1의 비율: 30%
stratify=Y로 설정 시, TestSet과 TrainSet에서 0의 비율과 1의 비율을 Data와 동일하게 유지
'''


'''
x_train, x_test = train_test_split(x, test_size=0.3)
y_train, y_test = train_test_split(y, test_size=0.3)
print("x_train, x_test: ", x_train, x_test, "\ny_train, y_test: ", y_train, y_test)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print("x_train, x_test: ", x_train, x_test, "\ny_train, y_test: ", y_train, y_test)

# Result
x_train, x_test:  [8 5 2 9 6 4 7] [ 3  1 10] 
y_train, y_test:  [2 1 7 9 5 4 0] [8 3 6]
x_train, x_test:  [9 5 2 8 7 4 1] [ 6 10  3]
y_train, y_test:  [8 4 1 7 6 3 0] [5 9 2]
'''


'''
numpy data 나누기(train:test)
train, test set의 범위를 편향되게 할 경우, train or test에 유불리한 결과가 나올 수 있음
그렇다고 시작과 끝을 맞춰서 같은 범위 내에서 돌아가게 할 필요는 X (시작과 끝의 데이터를 선택해서 넣게되는 문제가 발생하므로)
x_train = x[:7] # 시작 생략 가능
x_test = x[7:] # 끝 생략 가능
y_train = y[:7]
y_test = y[7:]
print(x_train, x_test, y_train, y_test)
'''


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