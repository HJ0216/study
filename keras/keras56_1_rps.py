from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    'D:/_data/rps',
    target_size=(300, 300),
    batch_size=2520,
    class_mode='categorical',
    color_mode='rgb',
    shuffle='True',
    )

print(xy_train[0][0].shape) # x: (2520, 300, 300, 3)
print(xy_train[0][1].shape) # y: (2520, 3)


# 2. Model
model = Sequential()
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(300,300,3)))
model.add(MaxPooling2D(pool_size=(10, 10)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(10, 10)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(10, 10)))

model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.summary()


# 3. Compile and Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

earlystop = EarlyStopping(monitor='val_acc', mode='max', patience=64,
                              restore_best_weights=True,
                              verbose=1)

hist = model.fit(xy_train[0][0], xy_train[0][1],
                    epochs=512,
                    batch_size=16,
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


# 5. Submission
test_filenames = os.listdir("D:/_data/dogs-vs-cats/test1")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]
'''
print(test_df)

        filename
0          1.jpg
1         10.jpg
2        100.jpg
3       1000.jpg
4      10000.jpg

print(test_df.shape[0]) # 12500
print(test_df.shape[1]) # 1

'''

test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "D:/_data/dogs-vs-cats/test1/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)
# generator 거쳐서 image shape이 변화 (128,128,)


# for submission
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

test_df['category'] = np.argmax(predict, axis=-1)
# axis=-1: 2차원일 때는 y축, 3차원일 때는 z축

label_map = dict((v,k) for k,v in train_generator.class_indices.items()) # train_generator의 class 확인
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })

'''
print(test_df)

        filename  category
0          1.jpg         1
1         10.jpg         0
2        100.jpg         1
3       1000.jpg         1
4      10000.jpg         1
'''

submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']

'''
print(submission_df)

        filename  category     id  label
0          1.jpg         1      1      1
1         10.jpg         0     10      0
2        100.jpg         1    100      1
3       1000.jpg         1   1000      1
4      10000.jpg         1  10000      1
...          ...       ...    ...    ...
12495   9995.jpg         1   9995      1
12496   9996.jpg         1   9996      1
12497   9997.jpg         1   9997      1
12498   9998.jpg         0   9998      0
12499   9999.jpg         1   9999      1
'''

submission_df.drop(['filename', 'category'], axis=1, inplace=True)
# drop: axis=1 -> col, inplace=True (변경된 값을 해당 df에 저장)

'''
print(submission_df)

          id  label
0          1      1
1         10      0
2        100      1
3       1000      1
4      10000      1
...      ...    ...
12495   9995      1
12496   9996      1
12497   9997      1
12498   9998      0
12499   9999      1
'''

submission_df.to_csv('submission.csv', index=False)
# df -> csv, idx=false -> idx를 내보내지 않음
