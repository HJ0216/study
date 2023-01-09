from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split



# 1. Data
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    shuffle= True,
    random_state = 333
)


# 2. Model Construction
model = Sequential()
# model.add(Dense(5, input_dim=13)) # (vector, scalar) input_dim 사용 가능
model.add(Dense(64, input_shape=(13,))) # 다차원의 경우 input_shape 사용
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))


# 3. Compile and train
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train,
          epochs=300,
          batch_size=16,
          validation_split=0.2,
          verbose=1)
# 함수 수행 시 발생하는 상세한 정보들을 표준 출력으로 자세히 내보낼 것인가를 나타냄
# 0: 미출력, 1(Default): 자세히, 2: 함축적 정보 출력 3. 2보다 더 함축적 정보 출력
# fit 수행 시, 수행 과정을 어떤식으로 보여주는가에 따라 소요시간이 변화함


# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("loss", loss)

print(hist) # <keras.callbacks.History object at 0x0000016CC0BBC4F0>
print(hist.history)
# fit의 history는 loss 결과값을 dictionary(key-value(list)) 형태로 반환
# key: 'loss', 'val_loss' / value = list[] 형태의 epochs의 loss, val_loss 값들의 집합
# {'loss': [25004.84765625, 1219.100830078125, 160.86378479003906, 82.83763122558594, 73.0763931274414, 71.3211669921875, 71.86249542236328, 70.77513885498047, 68.52639770507812, 68.08159637451172],
# 'val_loss': [2787.144775390625, 272.2074279785156, 78.57952880859375, 58.332862854003906, 55.535221099853516, 54.82481002807617, 54.39116668701172, 56.427764892578125, 59.47801971435547, 55.58904266357422]}

print(hist.history['loss']) # hist의 history 중 loss값만 반환
print(hist.history['val_loss']) # hist의 history 중 loss값만 반환


# (epochs, loss)의 산점도 및 그래프를 작성할 수 있음
import matplotlib.pyplot as plt

# plt.scatter(x, hist.history) # 산점도(x 생략 불가)
plt.figure(figsize=(9, 6)) # 그래프 사이즈 설정: figsize=(가로 inch, 세로 inch) 
plt.plot(hist.history['loss'], c='red', marker='.', label='loss') # x 추론 가능 시, x 생략 가능
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
# c: color, marker: graph 형태, label: graph name

plt.title('boston loss') # graph name
plt.xlabel('epochs') # x축 이름
plt.ylabel('loss') # x축 이름
plt.grid() # 격자 표시
plt.legend(loc='upper right') # 선 이름(label) 표시, location 미 지정 시 그래프와 안 겹치게 생성
# supported values are 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
plt.show()



'''
Result
plt.show()

'''