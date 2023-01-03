from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Data
x = np.array(range(1,21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    shuffle=True,
    random_state=123
)


# 2. Model Construction
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


# 3. compile and train
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=1)


# 4. Evalueate and Predict
loss = model.evaluate(x_test, y_test) # training이 안되어 있으므로 evalueate가 안좋음
print("Loss: ", loss)

y_predict = model.predict(x)
print("Result: ", y_predict)

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, y_predict, color="red")
plt.show()