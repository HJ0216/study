import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Data
train_datagen = ImageDataGenerator(
    rescale=1./255.,
    horizontal_flip=True, # 수평
    vertical_flip=True, # 수직
    width_shift_range=0.1, # 가로 이동
    height_shift_range=0.1, # 세로 이동
    rotation_range=5,# 훈련 시, 과적합 문제를 해결하기 위해 shift, ratatoin 시행
    zoom_range=1.2, # 20% 확대
    shear_range=0.7, # 절삭
    fill_mode='nearest' # 이동 시, 발생하는 빈 칸을 어떻게 채울 것인가
)

test_datagen = ImageDataGenerator(
    rescale=1./255.
)

xy_train = train_datagen.flow_from_directory(
    './_data/brain/train',
    target_size=(150, 150),
    batch_size=1000,
    class_mode='binary',
    color_mode='grayscale',
    shuffle='True', # ad, normal data shuffle
    )
# Found 160 images belonging to 2 classes.

xy_test = train_datagen.flow_from_directory(
    './_data/brain/test',
    target_size=(150, 150),
    batch_size=1000,
    class_mode='binary',
    color_mode='grayscale',
    shuffle='True',
    )
# Found 120 images belonging to 2 classes.
# x,y가 dic 형태로 들어가 있음

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000002134BCFCA60>

print(xy_train[0][0].shape)
print(xy_train[0][1].shape)


# 2. Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64,(2,2), input_shape=(150,150,1)))
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
hist = model.fit_generator(xy_train, steps_per_epoch=1, epochs=10,
                    validation_data=xy_test,
                    validation_steps=4) # x, y, batch_size 끌어오기, 10batch -> 12 훈련
# steps_per_epoch = total_data/batch_size
# 데이터 제너레이터를 사용하는 경우 검증 데이터의 배치를 끝없이 반환하므로 얼마나 많은 배치를 추출하여 평가할지 validation_steps 변수에 지정
# image data: gpu로 돌리기

accuracy = hist.history['acc']
val_acc = hist.history['val_acc']

loss = hist.history['loss']
val_loss = hist.history['val_loss']

print("Loss: ", loss[-1])
print("Val_Loss: ", val_loss[-1])
print("Accuracy: ", accuracy[-1])
print("Val_acc: ", val_acc[-1])


import matplotlib.pyplot as plt

xy_train_tmp = xy_train[0][0].reshape(160,22500)
xy_train_tmp2 = xy_train[0][1].reshape(160,1)

plt.plot(xy_train_tmp, xy_train_tmp2, color="gray")
plt.show()


