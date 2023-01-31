# 데이터를 np로 변환하여, 변환하는데 시간을 줄이고자 함
# 데이터를 변환한것보다 원본 상태 그대로 저장하고자 함

import numpy as np


# np.save('./_data/brain/brain_x_train.npy', arr=xy_train[0][0])
# np.save('./_data/brain/brain_y_train.npy', arr=xy_train[0][1])
# save -> numpy file로 저장

# np.save('./_data/brain/brain_xy_train.npy', array=xy_train[0])
# tuple 형태는 type: npy가 아니므로 npy로 저장 X

# np.save('./_data/brain/brain_x_test.npy', arr=xy_test[0][0])
# np.save('./_data/brain/brain_y_test.npy', arr=xy_test[0][1])
# np.save('./_data/brain/brain_xy_test.npy', array=xy_test[0])

x_train = np.load('./_data/brain/brain_x_train.npy')
y_train = np.load('./_data/brain/brain_y_train.npy')

x_test = np.load('./_data/brain/brain_x_test.npy')
y_test = np.load('./_data/brain/brain_y_test.npy')


# print(xy_train)
# print(xy_train[0][0].shape)
# print(xy_train[0][1].shape)
# print(type(xy_train))
# print(type(xy_train[0]))
# print(type(xy_train[0][0]))
# print(type(xy_train[0][1]))

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# matplotlib으로 확인 가능


# 2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64,(2,2), input_shape=(200,200,1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # result_y: 0 1
# softmax, 2


# 3. Compile and Train
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
# hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=10,
#                     validation_data=xy_test,
#                     validation_steps=4) # x, y, batch_size 끌어오기, 10batch -> 12 훈련
# steps_per_epoch = total_data/batch_size
# 데이터 제너레이터를 사용하는 경우 검증 데이터의 배치를 끝없이 반환하므로 얼마나 많은 배치를 추출하여 평가할지 validation_steps 변수에 지정
# image data: gpu로 돌리기

hist = model.fit(x_train, y_train,
                    batch_size=16,
                    # steps_per_epoch=10,
                    epochs=10,
                    validation_data=(x_test, y_test),
                    # validation_split: train data에서 나누기
                    # validation_steps=4
                    ) # x, y, batch_size 끌어오기, 10batch -> 12 훈련

accuracy = hist.history['acc']
val_acc = hist.history['val_acc']

loss = hist.history['loss']
val_loss = hist.history['val_loss']

print("Loss: ", loss[-1])
print("Val_Loss: ", val_loss[-1])
print("Accuracy: ", accuracy[-1])
print("Val_acc: ", val_acc[-1])



'''
Result
Loss:  0.028131157159805298
Val_Loss:  0.05063672363758087
Accuracy:  0.987500011920929
Val_acc:  0.9750000238418579

'''