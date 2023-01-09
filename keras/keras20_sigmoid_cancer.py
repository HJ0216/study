from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Data
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

# data: x, target: y(0 or 1)
x = datasets['data'] # x=datasets.data
y = datasets['target']
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    shuffle= True,
    random_state = 333
)


# 2. Model Construction
model = Sequential()
model.add(Dense(50, activation='linear', input_shape=(30,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 3. Compile and train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Binary Classification Model Construction
# Last Activation=sigmoid, loss='binary_crossentropy'
# 이진분류에서는 acc 사용이 용이함

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1)

model.fit(x_train, y_train,
          epochs=10000, batch_size=16,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)


# 4. evaluate and predict
loss, accuracy = model.evaluate(x_test, y_test) # return 값이 2개 이상일 경우, variable
print("loss: ", loss)
print("accuracy: ", accuracy)

y_predict = model.predict(x_test) # y_test와 비교
# print(y_predict[:10]) # 10개 출력, 0<sigmoid<1
# print(y_test[:10]) # 10개 출력 y_test = 0 or 1
# 실수와 정수 타입 언매칭(ValueError: Classification metrics can't handle a mix of binary and continuous targets)

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print("accuarcy_score: ", acc)


'''
Result
# loss = model.evaluate(x_test, y_test)
# print("[binarycrossentropy, accuracy]: ", loss)
Restoring model weights from the end of the best epoch: 32.
[binarycrossentropy, accuracy]:  [0.16636592149734497, 0.9561403393745422]

Updated Result
Restoring model weights from the end of the best epoch: 198.
loss:  0.16303640604019165
accuracy:  0.9561403393745422

sigmoid 이진 분류화

'''
