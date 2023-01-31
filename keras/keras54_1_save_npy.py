# 데이터를 np로 변환하여, 변환하는데 시간을 줄이고자 함
# 데이터를 변환한것보다 원본 상태 그대로 저장하고자 함

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255., # Scaling
    # horizontal_flip=True, # 수평 뒤집기
    # vertical_flip=True, # 수직 뒤집기
    # width_shift_range=0.1, # 가로 이동
    # height_shift_range=0.1, # 세로 이동
    # rotation_range=5,# 훈련 시, 과적합 문제를 해결하기 위해 shift, ratatoin 시행
    # zoom_range=1.2, # 20% 확대
    # shear_range=0.7, # 절삭
    # fill_mode='nearest' # 이동 시, 발생하는 빈 칸을 어떻게 채울 것인가
)

test_datagen = ImageDataGenerator(
    rescale=1./255.
)

# test는 정확한 평가 하기 위해 data를 처리하지 않음

xy_train = train_datagen.flow_from_directory(
    './_data/brain/train',
    target_size=(200, 200), # data 처리 시, 모든 데이터의 사이즈를 동일하게 맞춤
    batch_size=100000,
    class_mode='binary', # 폴더 라벨링 방식 지정: binary(0 1)
    color_mode='grayscale', # 색상: 흑백
    shuffle='True',
    )

xy_test = train_datagen.flow_from_directory(
    './_data/brain/test',
    target_size=(200, 200),
    batch_size=100000,
    class_mode='binary',
    color_mode='grayscale',
    shuffle='True',
    )
# Found 120 images belonging to 2 classes.

# y: 0 1로 이뤄진 scalar 집합

np.save('./_data/brain/brain_x_train.npy', arr=xy_train[0][0])
np.save('./_data/brain/brain_y_train.npy', arr=xy_train[0][1])
# np.save('./_data/brain/brain_xy_train.npy', array=xy_train[0])
# save -> numpy file로 저장

np.save('./_data/brain/brain_x_test.npy', arr=xy_test[0][0])
np.save('./_data/brain/brain_y_test.npy', arr=xy_test[0][1])
# np.save('./_data/brain/brain_xy_test.npy', array=xy_test[0])


# print(xy_train)
# print(xy_train[0][0].shape)
# print(xy_train[0][1].shape)
# print(type(xy_train))
# print(type(xy_train[0]))
# print(type(xy_train[0][0]))
# print(type(xy_train[0][1]))

print(xy_train[0][1])
'''
class_mode: binary
shape: (10,)
[1. 0. 1. 1. 0. 1. 0. 1. 0. 0.]

class_mode: one-hot type
shape: (10, 2)
[[0. 1.]
 [1. 0.]
 [0. 1.]
 [1. 0.]
 [0. 1.]
 [1. 0.]
 [1. 0.]
 [1. 0.]
 [0. 1.]
 [1. 0.]]
'''

