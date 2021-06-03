# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:54:34 2021

@author: rohan
"""

import tensorflow as tf
import json
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

with open('sarcasm.json','r') as f:
    datastore=json.load(f)

sentence=[]
label=[]
urls=[]

for item in datastore:
    sentence.append(item['headline'])
    label.append(item['is_sarcastic'])
    
training_sentences = sentence[0:training_size]
testing_sentences = sentence[training_size:]
training_labels = label[0:training_size]
testing_labels = label[training_size:]

tokenizer=Tokenizer(num_words=vocab_size,oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
testing_sequences=tokenizer.texts_to_sequences(testing_sentences)
pad=pad_sequences(training_sequences,truncating=trunc_type,padding=padding_type)
pade=pad_sequences(testing_sequences,truncating=trunc_type,padding=padding_type)
model=tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size,embedding_dim))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)))
model.add(tf.keras.layers.Dense(6,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
num_epochs = 50
training_padded = np.array(pad)
training_labels = np.array(training_labels)
testing_padded = np.array(pade)
testing_labels = np.array(testing_labels)
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=1)

