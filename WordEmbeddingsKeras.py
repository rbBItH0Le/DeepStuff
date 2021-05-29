# -*- coding: utf-8 -*-
"""
Created on Sat May 29 16:30:15 2021

@author: rohan
"""

import tensorflow as tf
import tensorflow_datasets as tdfs
import numpy as np

imdb,info=tdfs.load("imdb_reviews",with_info=True,as_supervised=True)

train_dataset=imdb['train']
test_dataset=imdb['test']

train_sentences=[]
train_labels=[]
test_sentences=[]
test_labels=[]

for s,l in train_dataset:
    train_sentences.append(str(s.numpy().decode('utf8')))
    train_labels.append(l.numpy())
    
for s,l in test_dataset:
    test_sentences.append(str(s.numpy().decode('utf8')))
    test_labels.append(l.numpy())
    
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size=10000
OOV_TOKEN="<OOV>"
max_length=120
embedding_dim=16
trunc_type='post'

tokenizer=Tokenizer(num_words=vocab_size,oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(train_sentences)
sequences=tokenizer.texts_to_sequences(train_sentences)
wordindex=tokenizer.word_index
pade=pad_sequences(sequences,maxlen=max_length,truncating=trunc_type)

testing_sequences=tokenizer.texts_to_sequences(test_sentences)
paded=pad_sequences(testing_sequences,maxlen=max_length,truncating=trunc_type)


reverse_word_index=dict([(value,key) for (key,value) in wordindex.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i,'?') for i in text])

print(decode_review(pade[3]))
print(train_sentences[3])


model=tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(6,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

training_labels_final = np.array(train_labels)
testing_labels_final = np.array(test_labels)
model.fit(pade, training_labels_final, epochs=10, validation_data=(paded, testing_labels_final))

e = model.layers[0]
weights = e.get_weights()[0]
