from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
# 이미지 = 2차원(2D) -> Conv2D 사용

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2),
                 input_shape=(5,5,1)))
# 이미지 픽셀 수(5*5) 1개(흑백) 3개(컬러RPG) = filter 수
# kernel_size = 이미지 블럭 생성(2*2)
# 1 layer 연산 후 5*5 -> 4*4 -> ... 연산량이 점점 줄어감
# filter = 10: 5*5 흑백 이미지 1장을 10장으로 늘림
# hyper-parameter tuning: filters, kernel_size, activation 등
model.add(Conv2D(filters=5, kernel_size=(2,2)))
# filter = 5: 4*4(블럭화 후 이미지 크기가 줄어듬) 흑백 이미지 1장을 5장으로 늘림
# filter: Dense에서의 hidden output layer
# 분류 타입으로 Dense를 이용하여 y값과 연결
# column화 시키기(3*3*5=45) 45개의 특성을 갖는 n개의 이미지로 교육

model.add(Flatten()) # 4차원 -> 2차원(3*3*5) -> 45
# Conv2D를 Flatten 처리하는 이유: 이미지를 수치화 했을 때, 행렬 값이 아닌 수치화
model.add(Dense(10))
# (임의) hidden 10 layer 처리
model.add(Dense(1)) # 이미지의 최종 수치화
model.summary()



'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 4, 4, 10)          50
                            연산에 따라 사이즈가 줄어감
 conv2d_1 (Conv2D)           (None, 3, 3, 5)           205
                            5 = filter 수
 flatten (Flatten)           (None, 45)                0

 dense (Dense)               (None, 10)                460

 dense_1 (Dense)             (None, 1)                 11
 =================================================================
'''
