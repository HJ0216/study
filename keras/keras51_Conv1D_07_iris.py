import pandas as pd
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout, Conv2D, Flatten, MaxPooling2D, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# 1. Data
dataset = load_iris()

x = dataset.data # for training
y = dataset.target # for predict

y=to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=123
)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler .transform(x_test)

print(x_train.shape, x_test.shape) # (105, 4) (45, 4)

x_train = x_train.reshape(105, 4, 1)
x_test = x_test.reshape(45, 4, 1)


# 2. Model Construction
model = Sequential()
model.add(Conv1D(128, 2, padding='same', input_shape=(4,1))) 
model.add(Conv1D(64, 2, padding='same')) 
model.add(Dropout(0.2))
model.add(Conv1D(32, 2, padding='same'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))


# 3. Compile and Training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlyStopping = EarlyStopping(monitor='accuracy', mode='max', patience=32,
                              restore_best_weights=True,
                              verbose=1)

model.fit(x_train, y_train,
          epochs=256,
          batch_size=32,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)


# 4. Evaluation and Prediction
result = model.evaluate(x_test, y_test)
print("loss: ", result)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("R2: ", r2)



'''
Result
loss:  [0.0711413100361824, 0.9555555582046509]
R2:  0.9330543095627365

'''