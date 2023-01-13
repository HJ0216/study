from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
# 이미지 = 2차원(2D) -> Conv2D 사용

model = Sequential()
# 입력: 데이터개수(무시), 가로(5), 세로(5), 색깔(1 or 3)
# 데이터의 개수는 중요하지 않으므로 (N, 5, 5, 1)
model.add(Conv2D(filters=10, kernel_size=(2,2),
                 input_shape=(5,5,1)))
# 이미지 픽셀 수(5*5) 1개(흑백) 3개(컬러RPG)
# kernel_size = 합을 연산할 이미지 블럭의 사이즈(2*2)
# 1 layer 연산 후 5*5 -> 4*4 -> ... 연산량이 점점 줄어감
# filter = 10: 5*5 흑백 이미지 1장을 10장으로 늘림
# hyper-parameter tuning: filters, kernel_size, activation 등

model.add(Conv2D(5, (2,2))) # model.add(Conv2D(filters=5, kernel_size=(2,2)))
# 2번째 conv2D는 1번째 모델의 output(4*4*10)
# filter = 5: 4*4(블럭화 후 이미지 크기가 줄어듬) 흑백 이미지 1장을 5장을 쌓아둠
# filter: Dense의 hidden output layer
# 이미지를 수치화 했을 때, column으로 class를 만들어서 y값과 연결: Conv2D->Flatten
model.add(Flatten()) # (3*3*5) -> (45, ) (Nan, 45)
# column화 시키기(3*3*5=45): 45개의 특성(Column)을 갖는 n개의 이미지로 교육

model.add(Dense(units=10))
# (임의) hidden 10 layer 처리
model.add(Dense(1)) # 이미지의 최종 수치화, (Nan, 1)

model.summary()



'''
cnn 연산 검색
'''

'''
Input shape
Conv2D(batch_size, rows, columns, channels(colors, filters))
-> batch_size: 훈련의 단위(10개의 데이터+2batch_size: 2개씩 5번)

Dense(units)
Input shape
Dense(batch_size, input_dim)
input_dim=col 개수
Dense(4, activation='relu')
'''


'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 4, 4, 10)          50 
                            연산에 따라 사이즈가 줄어감
 conv2d_1 (Conv2D)           (None, 3, 3, 5)           205
                            5 = filter 수
 flatten (Flatten)           (None, 45)                0 (flatten은 펴주는 것이므로 연산량 X) -------------------DNN---------------------

 dense (Dense)               (None, 10)                460(bias +1)

 dense_1 (Dense)             (None, 1)                 11
 =================================================================
 
 
'''
