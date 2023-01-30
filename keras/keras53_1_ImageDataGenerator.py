import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 이미지를 데이터로 변경하고 증폭시켜서 훈련을 시키는 역할을 하는 method

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

# test는 정확한 평가 하기 위해 data를 처리하지 않음
# test data로 증폭을 시킬 경우, 데이터 조작이 문제가 될 수 있음

xy_train = train_datagen.flow_from_directory(
    './_data/brain/train',
    target_size=(200, 200), # data 처리 시, 모든 데이터의 사이즈를 동일하게 맞춤
    batch_size=10, # 5개씩 잘라서 훈련, model.fit(batch_size) 사용 X, 1epoch=16번
    # batch_size를 최대로 올려서 데이터셋 범위 알아내기 -> [0][0]
    # print(xy_train[0][1].shape) # data_y: (5, ) -> (160, 200, 200, 1) (160, )
    class_mode='binary',
    color_mode='grayscale',
    shuffle='True',
    )
# dir = folder, folder에 image data 가져오기
# x: 160장, 150*150*1(흑백)


# y: 160장, 예측할 dataset: 160, 예측할 값이


# y.unique(): [0] - 80장, [1] - 80장
# parameter, 가장 마지막에 ','가 있어도 문제 X
# print() 구문없이도 Found 160 images belonging to 2 classes.

xy_test = train_datagen.flow_from_directory(
    './_data/brain/test',
    target_size=(200, 200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle='True',
    )
# Found 120 images belonging to 2 classes.

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000002134BCFCA60>
# print(xy_train[0])
# batch size만큼 y_class 출력
print(xy_train[0][0].shape) # data_x: (5, 200, 200, 1)
# print(xy_train[0][1]) # data_y: [1. 1. 1. 0. 1.]
print(xy_train[0][1].shape) # data_y: (5, )
print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) # <class 'tuple'> list는 변경이 가능하나, tuple은 변경 X
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>
