import numpy as np
import pandas as pd 
import random
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 1. Data
IMAGE_WIDTH=300
IMAGE_HEIGHT=300
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

print(os.listdir("D:/_data/rps"))
# ['paper', 'rock', 'scissors']

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=15,
    zoom_range=0.2,
    shear_range=0.1,
    fill_mode='nearest'
)

xy_train = train_datagen.flow_from_directory(
    'D:/_data/rps',
    target_size=IMAGE_SIZE,
    batch_size=2520,
    class_mode='categorical',
    color_mode='rgb',
    shuffle='True',
    )
# Found 2520 images belonging to 3 classes.

'''
x_train = np.load('D:/_data/rps_numpy/rps_x_train.npy')
y_train = np.load('D:/_data/rps_numpy/rps_y_train.npy')

print(x_train.shape) # x: (2520, 300, 300, 3)
print(y_train.shape) # y: (2520, 3)
'''

# 2. Model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(300,300,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()


# 3. Compile and Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

earlystop = EarlyStopping(monitor='val_acc', mode='max', patience=64,
                              restore_best_weights=True,
                              verbose=1)

hist = model.fit(xy_train[0][0], xy_train[0][1],
                    epochs=128,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=[earlystop],
                    verbose=1)


# 4. evaluate and predict
accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
print("Accuracy: ", accuracy[-1])
print("Val_acc: ", val_acc[-1])

loss = hist.history['loss']
val_loss = hist.history['val_loss']
print("Loss: ", loss[-1])
print("Val_Loss: ", val_loss[-1])



'''
Result
Accuracy:  0.9980158805847168
Val_acc:  0.9444444179534912

Loss:  0.01348116435110569
Val_Loss:  0.5943151116371155

'''


# 5. Submission
test_filenames = os.listdir("D:/_data/rps_test")
test_df = pd.DataFrame({
    'filename': test_filenames
})

print(test_df.head())
'''
print(test_df)

  filename
0    a.jpg
1    b.jpg
2    c.jpg
3    d.jpg

print(test_df.shape[0]) # 26
print(test_df.shape[1]) # 1
'''

test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "D:/_data/rps_test/", 
    x_col='filename',
    y_col=None,
    class_mode=None, # no targets are returned
    target_size=IMAGE_SIZE,
    batch_size=26,
    shuffle=False
)


# for submission
predict = model.predict_generator(test_generator, steps=1)

test_df['category'] = np.argmax(predict, axis=-1)
# axis=-1: 2차원일 때는 y축, 3차원일 때는 z축

# label_map = dict((v,k) for k,v in xy_train.class_indices.items())
# test_df['category'] = test_df['category'].replace(label_map)
# test_df['category'] = test_df['category'].replace({ 'scissors': 2, 'rock': 1, 'paper': 0 })

print(test_df.head())
'''
print(test_df)
  filename  category
0    a.jpg         1
1    b.jpg         0
2   c.jpeg         1
3    d.jpg         1
'''

submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']

print(submission_df.head())
'''
print(submission_df)

  filename  category id  label
0    a.jpg         1  a      1
1    b.jpg         0  b      0
2   c.jpeg         1  c      1
3    d.jpg         1  d      1
'''

submission_df.drop(['filename', 'category'], axis=1, inplace=True)
# drop: axis=1 -> col, inplace=True (변경된 값을 해당 df에 저장)


print(submission_df.head())
'''
print(submission_df)

  id  label
0  a      1
1  b      0
2  c      1
3  d      1
'''

submission_df.to_csv('submission3.csv', index=False)
# df -> csv, idx=false -> idx를 내보내지 않음
