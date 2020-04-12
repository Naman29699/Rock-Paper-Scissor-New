import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

train_dir = r"C:\Users\naman\Desktop\try\train"
test_dir = r"C:\Users\naman\Desktop\try\test"

train_rock_dir = r"C:\Users\naman\Desktop\try\train\rock"
train_paper_dir = r"C:\Users\naman\Desktop\try\train\paper"
train_scissor_dir = r"C:\Users\naman\Desktop\try\train\scissor"

test_rock_dir = r"C:\Users\naman\Desktop\try\test\rock"
test_paper_dir = r"C:\Users\naman\Desktop\try\test\paper"
test_scissor_dir = r"C:\Users\naman\Desktop\try\test\scissor"

num_rock_tr = len(os.listdir(train_rock_dir))
num_paper_tr = len(os.listdir(train_paper_dir))
num_scissor_tr = len(os.listdir(train_scissor_dir))

num_rock_test = len(os.listdir(test_rock_dir))
num_paper_test = len(os.listdir(test_paper_dir))
num_scissor_test = len(os.listdir(test_scissor_dir))

total_train = num_scissor_tr + num_paper_tr + num_rock_tr  
total_test = num_rock_test + num_paper_test + num_scissor_test

print('total training rock images:', num_rock_tr)
print('total training paper images:', num_paper_tr)
print('total training scissor images:', num_scissor_tr)

print('total test rock images:', num_rock_test)
print('total test paper images:', num_paper_test)
print('total test scissor images:', num_scissor_test)
print("--")
print("Total training images:", total_train)
print("Total test images:", total_test)

IMG_HEIGHT = 150
IMG_WIDTH = 150

image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5)

train_data_gen = image_gen_train.flow_from_directory(batch_size=5,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     color_mode = 'grayscale',
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical')

image_gen_test = ImageDataGenerator(rescale=1./255)

test_data_gen = image_gen_test.flow_from_directory(batch_size=5,
                                                 directory=test_dir,
                                                 color_mode = 'grayscale',
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='categorical')


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,1)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    epochs=10,
    validation_data=test_data_gen
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json) 
model.save_weights('model.h5')  









