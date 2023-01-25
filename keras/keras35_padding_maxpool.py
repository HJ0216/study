import numpy as np
import datetime

from tensorflow.keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D # 2D: 이미지
# Maxpooling: 연산이 아니기때문에 layer에서 삽입
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


path = './_save/'
filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
# (60000, 28, 28) (60000,)
# 행(28), 열(28), 흑백(1-생략)인 이미지 데이터 60000개
# scalar=1인 데이터 60000개
print(x_test.shape, y_test.shape) # CNN Conv2D 처리하기 위해 4D(Tensor)화

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True))
'''
y_train 다중 분류인지 데이터 특성 파악
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), # y_calss
array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64) y_class의 개수)
'''


# 2. Model
model = Sequential()
model.add(Conv2D(filters=128,
                 kernel_size=(3, 3),
                 padding='same',
                 input_shape=(28, 28, 1),
                 activation='relu'))
# Conv2D 후, result (27, 27, 128)
# output shape = input_shape - kernel_size +1 (Not using padding)
# padding 후, (None, 28, 28, 128)
'''
padding=valid
output_shape=(26,26,128)
padding=same
output_shape=(28,28,128)

# hyperParameterTuning
kernel_size
padding opt
-> 어떤것이 더 좋다고 할 수는 없고 훈련을 통해서 조정
'''
model.add(MaxPooling2D()) # (14, 14, 128) Parameter=0
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 padding='same')) # Conv2D 후, result (26, 26, 64)
# Sequential Model output->input이므로 입력값 작성 생략
model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 padding='same')) # Conv2D 후, result (25, 25, 64)
model.add(MaxPooling2D()) # (7, 7, 64) Parameter=0
model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 padding='same')) # Conv2D 후, result (25, 25, 64)
model.add(Flatten()) # input_dim=25*25*64=40000 = column
model.add(Dense(32, activation='relu'))  # 32- 임의
# 60000=batch_size(총 훈련 필요 대상), 40000=input_dim
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax')) # 10=y_class
model.summary()


# 3. Compile and train
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
# one-hot encoding 안했으므로, sparse
# one-hot encoding 후 (60000, 10=y_class)


earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=32, restore_best_weights=True, verbose=1)

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

modelCheckPoint = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                                   save_best_only=True,
                                   filepath=filepath + 'k34_1_' + date + '_' + filename)


model.fit(x_train, y_train, epochs=64, batch_size=512,
                    validation_split=0.2,
                    callbacks=[earlyStopping, modelCheckPoint],
                    verbose=1)

model.save(path+'keras34_1_mnist_save_model.h5') # 가중치 및 모델 세이브


# 4. evaluate and predict
result = model.evaluate(x_test, y_test)
print("loss: ", result[0])
print("acc: ", result[1])



'''
Result(2*2, with padding)
loss:  0.16121244430541992
acc:  0.9692999720573425

Result(3*3, with padding)
loss:  0.14477087557315826
acc:  0.9781000018119812

Result(2*2, without padding)
loss:  0.13609696924686432
acc:  0.9739999771118164

Result(3*3, without padding)
loss:  2.301159620285034
acc:  0.11349999904632568

Result(3*3, with padding, maxpooling)
loss:  0.04873143881559372
acc:  0.9861000180244446

'''