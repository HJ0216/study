import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# 1. Data
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))

'''
(array([1, 2, 3, 4, 5, 6, 7]),
array([211840, 283301,  35754,   2747,   9493,  17367,  20510],
dtype=int64))

print(datasets.DESCR) # pandas.describe / cvs_type.info
print(datasets.feature_names) # pandas.columns, column name
'''



'''
Result
y=to_categorical(y)
print(y.shape) # (581012, 8)

to_catergorical: class가 0부터 시작하지 않을 때, 앞에 0을 추가 -> 추가된 0만큼의 자원의 낭비가 발생
to_categorical: (0,1,2,3,4,5,6,7)
y: (1,2,3,4,5,6,7)


y = pd.get_dummies(y)
'''



onehot_encoder=OneHotEncoder(sparse=False)
# return one-hot numeric an array
# Encode categorical features as a one-hot numeric array.
# sparse : bool, default=True
# Will return sparse matrix if set True else will return an array.
print("Original y ", y)
# y = datasets['target']
# [5 5 2 ... 3 3 3]
# print(y.shape) (581012,)
reshaped=y.reshape(len(y), 1) # (581012, 1): vector에 담아줌
# len(y): 581012 = data의 개수 -> reshaped.shape (581012, 1)
print("Before onehot_encoder y: ", y)
# before y:  [5 5 2 ... 3 3 3]
y=onehot_encoder.fit_transform(reshaped)

'''
print("After onehot_encoder y: ", y)
After onehot_encoder y:
[[0. 0. 0. ... 1. 0. 0.]
 [0. 0. 0. ... 1. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]]
-> onehot type(class -> column으로 변환하여 값을 도출)

'''

print(y.shape) #(581012, 7)
# transformed array.shape (n_samples, n_features_new=encoding에 따라 class-> col)


x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    shuffle=True,
    random_state=333,
    test_size=0.2,
    stratify=y
)


# 2. Model Construction
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(54, )))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='linear'))
model.add(Dense(7,activation='softmax'))


# 3. Compile and train
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=128,
          validation_split=0.2,
          verbose=1)


# 4. evaluate and predict
loss, accuracy = model.evaluate(x_test, y_test)
print("loss: ", loss)
print("accuracy: ", accuracy)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_predict)
print("accuracy_score: ", acc)


'''
Result

loss:  0.596627414226532
accuracy:  0.7471494078636169
accuracy_score:  0.7471493851277505
'''