import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Data
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(y)

print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))

y=to_categorical(y)
y = np.delete(y, 0, 1)

'''
print(y.shape) # (581012, 7)

print(y)
[[0. 0. 0. ... 1. 0. 0.]
 [0. 0. 0. ... 1. 0. 0.]
 [0. 1. 0. ... 0. 0. 0.]
 ...
 [0. 0. 1. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]]
 
'''

'''
Problem
to_catergorical: class가 0부터 시작하지 않을 때, 앞에 0을 추가 -> 추가된 0만큼의 자원의 낭비가 발생
to_categorical(y): (0,1,2,3,4,5,6,7)
y: (1,2,3,4,5,6,7)

Solution
y = np.delete(y, 0, 1)
np.delete(array, idx, axis)
numpy 배열의 첫번째 열을 삭제

'''

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    shuffle=True,
    random_state=333,
    test_size=0.2,
    stratify=y
)


# 2. Model Construction
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(54, )))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='linear'))
model.add(Dense(7,activation='softmax'))


# 3. Compile and train
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=128,
          validation_split=0.2,
          verbose=1)


# 4. evaluate and predict
loss, accuracy = model.evaluate(x_test, y_test)
print("loss: ", loss)
print("accuracy: ", accuracy)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_predict)
print("accuracy_score: ", acc)


'''
Result

loss:  0.6315832138061523
accuracy:  0.7298176288604736
accuracy_score:  0.729817646704474

'''