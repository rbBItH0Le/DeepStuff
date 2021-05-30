# -*- coding: utf-8 -*-
"""
Created on Sun May 30 17:04:24 2021

@author: rohan
"""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
imdb,info=tfds.load("imdb_reviews/subwords8k",as_supervised=True,with_info=True)

train_set=imdb['train']
test_set=imdb['test']

train_sente=[]
test_sente=[]

trains_label=[]
tests_label=[]
for s,l in train_set:
    train_sente.append((s.numpy()))
    trains_label.append(l.numpy())
    
for s,l in test_set:
    test_sente.append(s.numpy())
    tests_label.append(l.numpy())
    
trains_label=np.array(trains_label)
tests_label=np.array(tests_label)
    
tokenizer=info.features['text'].encoder
sub=tokenizer.subwords

sample_string = 'TensorFlow, from basics to mastery'
tokenized_string=tokenizer.encode(sample_string)
detokenized_string=tokenizer.decode(tokenized_string)

for ts in tokenized_string:
    print("{}---{}".format(ts,tokenizer.decode([ts])))
    
from tensorflow.keras.preprocessing.sequence import pad_sequences
pade=pad_sequences(train_sente,maxlen=150,truncating='post')
paded=pad_sequences(test_sente,maxlen=150,truncating='post')   
    

embedding_dim=64
model=tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(tokenizer.vocab_size,embedding_dim))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(6,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


model.fit(pade,trains_label,epochs=10,validation_data=(paded,tests_label))
