# -*- coding: utf-8 -*-
"""
Created on Sat May 22 20:16:44 2021

@author: rohan
"""
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.applications.inception_v3 import InceptionV3
loaded_weights='inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model=InceptionV3(include_top=False,input_shape=(150,150,3),weights=None)
pre_trained_model.load_weights(loaded_weights)


for layer in pre_trained_model.layers:
    layer.trainable=False
    
last_layer=pre_trained_model.get_layer('mixed7')

last_layer.output_shape
outputs=last_layer.output

x=layers.Flatten()(outputs)
x=layers.Dense(1024,activation='relu')(x)
x=layers.Dropout(0.2)(x)
x=layers.Dense(1,activation='sigmoid')(x)

model=Model(pre_trained_model.input,x)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

traind=ImageDataGenerator(rescale=1/255,rotation_range=40,shear_range=0.2,height_shift_range=0.2,width_shift_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
testd=ImageDataGenerator(rescale=1/255)

train_data_generator=traind.flow_from_directory('cats-v-dogs/train/',target_size=(150,150),batch_size=20,class_mode='binary')
valid_data_generator=testd.flow_from_directory('cats-v-dogs/validation/',target_size=(150,150),batch_size=20,class_mode='binary')

from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_data_generator,epochs=20,steps_per_epoch=100,validation_data=valid_data_generator,verbose=2)


test_datagen = ImageDataGenerator(rescale=1/255)
TestDataGene=test_datagen.flow_from_directory('cats-v-dogs/test/',target_size=(150,150),shuffle=False,batch_size=20)