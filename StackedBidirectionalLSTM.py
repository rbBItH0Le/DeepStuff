# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:21:42 2021

@author: rohan
"""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
imdb,info=tfds.load("imdb_reviews/subwords8k",with_info=True,as_supervised=True)

train_set=imdb['train']
test_set=imdb['test']

tokenizer=info.features['text'].encoder

from tensorflow.keras.preprocessing.sequence import pad_sequences
train_sentences=[]
train_labels=[]

test_sentences=[]
test_labels=[]
for s,l in train_set:
    train_sentences.append(s.numpy())
    train_labels.append(l.numpy())

for s,l in test_set:
    test_sentences.append(s.numpy())
    test_labels.append(l.numpy())
    
train_labels=np.array(train_labels)
test_labels=np.array(test_labels)

pade=pad_sequences(train_sentences,maxlen=150,truncating='post')
padedd=pad_sequences(test_sentences,maxlen=150,truncating='post')


model=tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(tokenizer.vocab_size,64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(pade,train_labels,epochs=10,validation_data=(padedd,test_labels))