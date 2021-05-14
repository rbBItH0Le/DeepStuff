# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:32:37 2021

@author: rohan
"""
os.chdir('C:\\Users\\rohan\\Documents')
import os,zipfile
localzip='dogs-vs-cats.zip'
zipfile_ref=zipfile.ZipFile(localzip,'r')
zipfile_ref.extractall()
base_dir = 'dogs-vs-cats/'
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')

train_dogs_dir=os.path.join(train_dir,'dogs')
train_cats_dir=os.path.join(train_dir,'cats')

validation_dogs_dir=os.path.join(validation_dir,'dogs')
validation_cats_dir=os.path.join(validation_dir,'cats')

train_dog_names=os.listdir(train_dogs_dir)
train_cats_names=os.listdir(train_cats_dir)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
n_rows=4
n_columns=4
fig=plt.gcf()

fig.set_size_inches(4*n_rows,4*n_columns)
next_cat_pics=[os.path.join(train_cats_dir,fn) for fn in train_cats_names[0:8]]
next_dog_pics=[os.path.join(train_dogs_dir,fn) for fn in train_dog_names[0:8]]

for i,img_path in enumerate(next_cat_pics+next_dog_pics):
    sp=plt.subplot(n_rows,n_columns,i+1)
    sp.axis('Off')
    img=mpimg.imread(img_path)
    plt.imshow(img)
plt.show()

import tensorflow as tf
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator


traindatagen=ImageDataGenerator(rescale=1/255)
validdatagen=ImageDataGenerator(rescale=1/255)

TrainDataGene=traindatagen.flow_from_directory('dogs-vs-cats/train',target_size=(200,200),class_mode='binary',batch_size=10)
ValidDataGene=validdatagen.flow_from_directory('dogs-vs-cats/validation',target_size=(200,200),class_mode='binary',batch_size=10)

model.fit(TrainDataGene,epochs=10,validation_data=ValidDataGene,verbose=2)

test_datagen = ImageDataGenerator(rescale=1/255)
TestDataGene=test_datagen.flow_from_directory('dogs-vs-cats/test',target_size=(200,200),class_mode='binary',batch_size=20,shuffle=False)
len(TestDataGene.filenames)

pred=model.predict_generator(TestDataGene, steps=len(TestDataGene), verbose=1)
import numpy as np
cl = np.round(pred)
import pandas as pd
