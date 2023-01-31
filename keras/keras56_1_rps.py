import numpy as np
import pandas as pd 
import random
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from sklearn.model_selection import train_test_split


# 1. Data
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

xy_train = train_datagen.flow_from_directory(
    './_data/rps',
    target_size=(300, 300),
    batch_size=840,
    class_mode='categorical',
    color_mode='rgb',
    shuffle='True',
    )

# xy_train[0][0] = np.load('D:/_data/rps/rps_x_train.npy')
# xy_train[0][1] = np.load('D:/_data/rps/rps_y_train.npy')


print(xy_train[0][0].shape) # x: (20, 300, 300, 3)
print(xy_train[0][1].shape) # y: (20, 3)


# 2. Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(300,300,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))


# 3. Compile and Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

earlystop = EarlyStopping(monitor='val_acc', mode='max', patience=64,
                              restore_best_weights=True,
                              verbose=1)

hist = model.fit(xy_train[0][0], xy_train[0][1],
                    epochs=512,
                    batch_size=8,
                    validation_split=0.2,
                    callbacks=[earlystop],
                    verbose=1)


# 4. evaluate and predict
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
Loss:  9.018582932185382e-05
Val_Loss:  1.3695333003997803
Accuracy:  1.0
Val_acc:  0.8690476417541504

'''