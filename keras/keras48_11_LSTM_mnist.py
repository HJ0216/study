import numpy as np

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist, cifar10, cifar100

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# 1. Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
'''
x_train.shape: (60000, 784), x_train.shape: (60000,)
x_test.shape: (10000, 784), x_test.shape: (10000,)
'''

print(x_train.shape, x_test.shape)
# (60000, 28, 28) (10000, 28, 28)

# x_train = x_train.reshape(60000, 784, 1)
# x_test = x_test.reshape(10000, 784, 1)

x_train=x_train/255.
x_test=x_test/255.


# 2. Model Construction
model = Sequential()
model.add(LSTM(units=64, input_shape=(28,28)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


# 3. Compile and Training
model.compile(loss='mse', optimizer='adam')

earlyStopping = EarlyStopping(monitor='loss', mode='min', patience=32, restore_best_weights=True, verbose=1)

model.fit(x_train, y_train, epochs=2, callbacks=[earlyStopping], batch_size=512)


# 4. Evaluation and Prediction
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2: ", r2)


'''
Result(epoch 수정하기)
RMSE:  1.5004572
R2:  0.7315063669285975

'''