import numpy as np
import pandas as pd
import datetime

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


path = './_save/'
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'


# 1. Data
datasets = fetch_covtype()

x = datasets.data
y = datasets['target']

y = y.reshape(581012, 1)

ohe = OneHotEncoder()
ohe.fit(y) # ohe에 y를 fit한 가중치를 넣음
y = ohe.transform(y) # 가중치가 들어간 ohe에 transform y 실행
y = y.toarray()


x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    shuffle=True,
    random_state=333,
    test_size=0.2,
    stratify=y
)

# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. Model Construction
input1 = Input(shape=(54,))
dense1 = Dense(64, activation='relu')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(64, activation='linear')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(32, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(16, activation='relu')(drop3)
output1 = Dense(7, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)


# 3. Compile and train
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

modelCheckPoint = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                                   save_best_only=True,
                                   filepath=filepath + 'k31_10_ohe_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=256, batch_size=128,
          validation_split=0.2,
          callbacks = [earlyStopping, modelCheckPoint],
          verbose=1)

model.save(path+'keras31_dropout10_ohe_save_model.h5') # 가중치 및 모델 세이브


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
loss:  0.596627414226532
accuracy:  0.7471494078636169
accuracy_score:  0.7471493851277505

Result using MinMaxScaler
loss:  0.28087565302848816
accuracy:  0.8855881690979004
accuracy_score:  0.8855881517688872

Result using Function
loss:  0.28988906741142273
accuracy:  0.8820512294769287
accuracy_score:  0.8820512379198472

Result using Drpoout
loss:  0.46084555983543396
accuracy:  0.8045231103897095
accuracy_score:  0.8045231190244658

'''
