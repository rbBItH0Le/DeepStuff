# -*- coding: utf-8 -*-


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)  
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

def windowed_dataset(series,batch_size,shuffle_buffer_size,window_size):
    series=tf.expand_dims(series,axis=-1)
    df=tf.data.Dataset.from_tensor_slices(series)
    df=df.window(window_size+1,shift=1,drop_remainder=True)
    df=df.flat_map(lambda window:window.batch(window_size+1))
    df=df.shuffle(shuffle_buffer_size).map(lambda window:(window[:-1],window[-1]))
    df=df.batch(batch_size).prefetch(1)
    for x,y in df:
        print('x=',x.numpy().shape)
        print('y=',y.numpy().shape)
    return df


train_set=windowed_dataset(series,32,1000,20)
model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=50,kernel_size=4,activation='relu',padding="causal",strides=1,input_shape=[None,1]))
model.add(tf.keras.layers.Conv1D(filters=50,kernel_size=4,activation='relu',padding="causal",strides=1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50,return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50)))
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Lambda(lambda x:x*100))
model.summary()
lr_schedule=tf.keras.callbacks.LearningRateScheduler(lambda epoch:1e-8*10**(epoch/20))
optimizer=tf.keras.optimizers.SGD(lr=9e-6,momentum=0.7)
model.compile(loss=tf.keras.losses.Huber(),optimizer=optimizer,metrics=["mae"])

history=model.fit(train_set,epochs=50)

plt.semilogx(history.history['lr'],history.history['loss'])

forecast=[]
for time in range(len(series)-window_size):
    serie1=series[time:time+window_size][np.newaxis]
    serie1=tf.expand_dims(serie1,axis=-1)
    forecast.append(model.predict(serie1))
    
forecast=forecast[split_time-window_size:]

results=np.array(forecast)[:,0,0]
    
plot_series(time_valid, x_valid)
plot_series(time_valid, results)


tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
