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
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission = pd.read_csv(path+'sampleSubmission.csv', index_col=0)
train_csv = train_csv.dropna()
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x.shape) # [10886 rows x 8 columns] -> dropna && drop로 인한 변경

y = train_csv['count']
print(y.shape) # Length: 10886


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    shuffle=True,
    train_size=0.7,
    random_state=123
)

print(x_train.shape, x_test.shape) # (7620, 8), (3266, 8)
print(y_train.shape, y_test.shape) # (7620,), (3266,)


# 2. Model Construction
model = Sequential()
model.add(Dense(64, input_shape=(8,)))
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


# 4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("loss", loss)

print(hist)
print(hist.history)
print(hist.history['loss'])
print(hist.history['val_loss'])


# (epochs, loss)의 산점도 및 그래프를 작성할 수 있음
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6)) 
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')

plt.title('Kaggle Bike loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.legend(loc='upper right')
plt.show()



'''
Result


'''