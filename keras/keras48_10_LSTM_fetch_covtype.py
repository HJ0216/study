import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# 1. Data
dataset = fetch_covtype()
x = dataset.data # for training
y = dataset.target # for predict

y = to_categorical(y)
y = np.delete(y, 0, axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=123
)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)

x_train = x_train.reshape(406708, 54, 1)
x_test = x_test.reshape(174304, 54, 1)


# 2. Model Construction
model = Sequential()
model.add(LSTM(units=64, input_shape=(54,1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


# 3. Compile and Training
model.compile(loss='mse', optimizer='adam')

earlyStopping = EarlyStopping(monitor='loss', mode='min', patience=32, restore_best_weights=True, verbose=1)

model.fit(x, y, epochs=2, callbacks=[earlyStopping], batch_size=1024)


# 4. Evaluation and Prediction
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2: ", r2)


'''
Result(epoch 수정하기), (4,2) 안돌아가는 이유 찾기

'''