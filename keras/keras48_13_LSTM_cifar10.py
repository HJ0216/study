import numpy as np

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# 1. Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)

x_train = x_train.reshape(50000, 32, 96)
x_test = x_test.reshape(10000, 32, 96)

x_train=x_train/255.
x_test=x_test/255.

'''
print(np.unique(y_train, return_counts=True))
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))
'''


# 2. Model Construction
model = Sequential()
model.add(LSTM(units=64, input_shape=(32,96)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))


# 3. Compile and Training
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=32, restore_best_weights=True, verbose=1)

model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=128,
          callbacks=[earlyStopping],
          batch_size=256)


# 4. Evaluation and Prediction
result = model.evaluate(x_test, y_test)
print("loss: ", result[0])
print("acc: ", result[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)

r2 = r2_score(y_test, y_predict)
print("R2: ", r2)



'''
Result
loss:  1.2765038013458252
acc:  0.5478000044822693
R2:  0.0021939393939394147

'''