import numpy as np
import pandas as pd 
import random
import os

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt


FAST_RUN = False #
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

filenames = os.listdir("D:/_data/dogs-vs-cats/train/")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1) # dog -> 1 추가
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

print(df.head())
'''
       filename  category
0     cat.0.jpg         0
1     cat.1.jpg         0
2    cat.10.jpg         0
3   cat.100.jpg         0
4  cat.1000.jpg         0
'''

sample = random.choice(filenames)
image = load_img("D:/_data/dogs-vs-cats/train/"+sample)
plt.imshow(image)
plt.show()

df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "D:/_data/dogs-vs-cats/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "D:/_data/dogs-vs-cats/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "D:/_data/dogs-vs-cats/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)

# print(type(validation_generator)) # <class 'keras.preprocessing.image.DataFrameIterator'>
print(type(train_generator[0][0])) # <class 'numpy.ndarray'>
print(type(train_generator[0][1])) # <class 'numpy.ndarray'>
print(train_generator[0][0].shape) # (15, 128, 128, 3)
print(train_generator[0][1].shape) # (15, 2):  one hot encoding type: class -> col
'''
# ImageDataGenerator Processing
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()

'''

np.save('D:/_data/dogs-vs-cats/train/cat_dog_x_train.npy', arr=train_generator[0][0])
np.save('D:/_data/dogs-vs-cats/train/cat_dog_y_train.npy', arr=train_generator[0][1])

np.save('D:/_data/dogs-vs-cats/train/cat_dog_x_val.npy', arr=validation_generator[0][0])
np.save('D:/_data/dogs-vs-cats/train/cat_dog_y_val.npy', arr=validation_generator[0][1])

# save -> numpy file로 저장
