import numpy as np
import datetime

from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D # 2D: 이미지
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


path = './_save/'
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
'''
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)
'''

# DNN Model을 위한 작업
x_train = x_train.reshape(50000, 32*32*3) # color=3, 해당 값도 계산하기
x_test = x_test.reshape(10000, 32*32*3)

x_train=x_train/255.
x_test=x_test/255.


# 2. Model
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(32*32*3,)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='linear'))
model.add(Dense(32, activation='linear'))
model.add(Dense(100, activation='softmax'))
# model.summary()


# 3. Compile and train
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=32, restore_best_weights=True, verbose=1)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

modelCheckPoint = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                                   save_best_only=True,
                                   filepath=filepath + 'k36_dnn4_' + date + '_' + filename)


model.fit(x_train, y_train, epochs=512, batch_size=256,
                    validation_split=0.2,
                    callbacks=[earlyStopping, modelCheckPoint],
                    verbose=1)

model.save(path+'keras36_dnn4_save_model.h5') # 가중치 및 모델 세이브


# 4. evaluate and predict
result = model.evaluate(x_test, y_test)
print("loss: ", result[0])
print("acc: ", result[1])



'''
Result(CNN)
loss:  2.973147392272949
acc:  0.2648000121116638

Result(DNN)
loss:  3.6616432666778564
acc:  0.14839999377727509

'''