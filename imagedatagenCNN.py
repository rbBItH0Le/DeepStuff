# -*- coding: utf-8 -*-
"""
Created on Thu May  6 22:52:42 2021

@author: rohan
"""

import os
import zipfile

local_zip= 'C:\\Users\\rohan\\Downloads\\horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('horse-or-human')
zip_ref.close()

train_horse_dir = os.path.join('horse-or-human/horses')
train_human_dir = os.path.join('horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nrows = 4
ncols = 4
pic_index = 0
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)
pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname) 
                for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) 
                for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

import tensorflow as tf
model1=tf.keras.models.Sequential()
model1.add(tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)))
model1.add(tf.keras.layers.MaxPool2D((2,2)))
model1.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu'))
model1.add(tf.keras.layers.MaxPool2D((2,2)))
model1.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model1.add(tf.keras.layers.MaxPool2D((2,2)))
model1.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model1.add(tf.keras.layers.MaxPool2D((2,2)))
model1.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model1.add(tf.keras.layers.MaxPool2D((2,2)))
model1.add(tf.keras.layers.Flatten())
model1.add(tf.keras.layers.Dense(512,activation='relu'))
model1.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model1.summary()

from tensorflow.keras.optimizers import RMSprop
model1.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator
traindatagen=ImageDataGenerator(rescale=1/255)
train_generator=traindatagen.flow_from_directory('horse-or-human/',class_mode='binary',batch_size=128,target_size=(300,300))
model1.fit(train_generator,epochs=15,steps_per_epoch=8,verbose=1)


uploaded=['BoJack_Horseman_Season_Six_design.png','images.jfif','Mzc5Njk1Ng.jpeg','hu.jpg','human02-20.png','gal.jfif','rea.png']
from tensorflow.keras.preprocessing import image
import numpy as np
for fn in uploaded:
 
  # predicting images
  path = 'C:\\Users\\rohan\\Documents\\testset\\' + fn
  img = image.load_img(path, target_size=(300, 300))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])
  classes = model1.predict(images, batch_size=10)
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")
    
import random
successive_outputs = [layer.output for layer in model1.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model1.input, outputs = successive_outputs)
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files)
img = image.load_img(img_path, target_size=(300, 300))
x =image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x/=255
successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in model1.layers[1:]]
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:
        n_features = feature_map.shape[-1]
        size = feature_map.shape[1]
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, i * size : (i + 1) * size] = x
        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')