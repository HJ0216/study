import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Data
train_datagen = ImageDataGenerator(
    rescale=1./255.,
    horizontal_flip=True, # 수평 뒤집기
    vertical_flip=True, # 수직 뒤집기
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
    target_size=(200, 200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle='True', # ad, normal data shuffle
    )
# Found 160 images belonging to 2 classes.

xy_test = train_datagen.flow_from_directory(
    './_data/brain/test',
    target_size=(200, 200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle='True',
    )
# Found 120 images belonging to 2 classes.
# x,y가 dictionary 형태로 들어가 있음

print(xy_train) # <keras.preprocessing.image.DirectoryIterator object at 0x000002134BCFCA60>

print(xy_train[0][0].shape) # (10 ,200, 200, 1)
print(xy_train[0][1].shape) # (10,)


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
model.add(Dense(1, activation='sigmoid'))
# model.add(Dense(2, activation='softmax')) # class_y: 0 1
# one_hot_encoding X: model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# one_hot_encoding O: model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])


# 3. Compile and Train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=128,
                    validation_data=xy_test,
                    validation_steps=4)
# fit_generator: x, y, batch_size 참조
# steps_per_epoch = total_data/batch_size
# validation_steps: validation data scale/batch_size

accuracy = hist.history['acc']
val_acc = hist.history['val_acc']

loss = hist.history['loss']
val_loss = hist.history['val_loss']

print("Loss: ", loss)


print("Loss: ", loss[-1])
print("Val_Loss: ", val_loss[-1])
print("Accuracy: ", accuracy[-1])
print("Val_acc: ", val_acc[-1])

'''
Result
Loss:
[0.7572494745254517, 0.7106074094772339, 0.6944756507873535, 0.693257749080658, 0.6952861547470093,
0.7542158961296082, 0.6947949528694153, 0.6976414918899536, 0.6793937683105469, 0.7301629781723022,
0.693245530128479, 0.693304181098938, 0.6932340860366821, 0.6932304501533508, 0.6932072043418884,
0.6931979656219482, 0.6932152509689331, 0.6932642459869385, 0.6932310461997986, 0.6931989192962646,
0.693191409111023, 0.6931725740432739, 0.6932080984115601, 0.6931743621826172, 0.693291425704956,
0.6933425664901733, 0.6932045221328735, 0.6932088136672974, 0.6931614875793457, 0.6932255029678345,
0.6931887865066528, 0.6932308077812195, 0.6933176517486572, 0.6932500600814819, 0.6931840777397156,
0.6932026147842407, 0.6931670308113098, 0.6931958198547363, 0.6931946873664856, 0.6932119131088257,
0.6932288408279419, 0.6931642293930054, 0.6933351755142212, 0.6931711435317993, 0.6931785345077515,
0.6931710839271545, 0.6931647658348083, 0.6931780576705933, 0.6932647824287415, 0.6931532621383667,
0.693471372127533, 0.6931419968605042, 0.6931999921798706, 0.6932162046432495, 0.6931958794593811,
0.6931720972061157, 0.6932581663131714, 0.6932448148727417, 0.693221926689148, 0.6931708455085754,
0.6932061910629272, 0.693199634552002, 0.6931688189506531, 0.6932955384254456, 0.693212628364563,
0.693221390247345, 0.6931605935096741, 0.6931674480438232, 0.6932255625724792, 0.6931595802307129,
0.693195641040802, 0.6931711435317993, 0.6932152509689331, 0.6932026743888855, 0.6933947205543518,
0.6931663751602173, 0.6932100057601929, 0.6932055950164795, 0.6932298541069031, 0.6931821703910828,
0.693175196647644, 0.6932721138000488, 0.6933510899543762, 0.6931986808776855, 0.6932271122932434,
0.6931864023208618, 0.6931711435317993, 0.6932213306427002, 0.6931648254394531, 0.6932080984115601,
0.6931921243667603, 0.6931769251823425, 0.6931687593460083, 0.6932806372642517, 0.6931632161140442,
0.6931666135787964, 0.6932119727134705, 0.6932608485221863, 0.6931592226028442, 0.6931537985801697,
0.6931866407394409, 0.6931837201118469, 0.6931716799736023, 0.6932442784309387, 0.6932657361030579,
0.6932154297828674, 0.6933247447013855, 0.6931847333908081, 0.6931852698326111, 0.6931909322738647,
0.6932597756385803, 0.6931959390640259, 0.6932161450386047, 0.6932095289230347, 0.6933270692825317,
0.6931805610656738, 0.6931859850883484, 0.6932801008224487, 0.6932880282402039, 0.6932093501091003,
0.6932862997055054, 0.6931726932525635, 0.6931778788566589, 0.693168044090271, 0.6931873559951782,
0.6931617856025696, 0.6931730508804321, 0.6932439804077148]

'''


import matplotlib.pyplot as plt

img = xy_train[0] # 1 batch(10개의 image set)을 img에 저장

plt.figure(figsize=(20, 10))
for i, img in enumerate(img[0]): # enumerate: (index, list_element)를 tuple type으로 반환
    # enumerate(img[0][0])
    # 루프가 반복될 때마다 변수 i는 현재 요소의 인덱스로 업데이트되고, img는 현재 요소의 값으로 업데이트 됨
    plt.subplot(1, 10, i+1) # subplot(row, col, Index 지정: 1, 2, ...): 전체 이미지 내에 포함된 내부 이미지 개수
    plt.axis('off')
    plt.imshow(img.squeeze()) # 차원(axis) 중, size가 1 인것을 찾아 스칼라 값으로 바꿔 해당차원을 제거
'''
squeeze()
x3: array([[[0]],
           [[1]],
           [[2]],
           [[3]],
           [[4]],
           [[5]]])
x3.shape: (6,1,1)

x3.squeeze()
array([0, 1, 2, 3, 4, 5])
'''
plt.tight_layout()
plt.show()
