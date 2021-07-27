# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
import pickle

import os

os.environ['PYTHONHASHSEED'] = str(42)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

tf.random.set_seed(42)
np.random.seed(42)
rn.seed(42)
# %%
train_dir = './data/X_train.pickle'
train_label_dir = './data/data_y_train.csv'
test_dir = './data/X_test.pickle'
test_label_dir = './data/data_y_test.csv'

with open(train_dir,'rb') as frb:
    X_train = pickle.load(frb) # train_feature_load

with open(test_dir,'rb') as frb:
    X_test = pickle.load(frb) # train_feature_load

X_train = np.array(X_train)[:,:,2:]
X_test = np.array(X_test)[:,:,2:]
y_train = pd.read_csv(train_label_dir)['label']  # train_label load
y_test = pd.read_csv(test_label_dir)['label']  # train_label load
# %%
from tensorflow.keras import layers, initializers
from tensorflow.keras.models import Model
# %%
filter_1 = 19
filter_2 = 19
filter_3 = 43
channel_1 = 128
channel_2 = 256
channel_3 = 128
input_channel = 8
stride = 2
drop_rate = 0.3

inputs = layers.Input(shape=(600,input_channel))

conv = layers.Conv1D(
    channel_1,filter_1,strides=stride,padding='same',activation='relu',
    kernel_initializer=initializers.he_normal(seed=42))(inputs)
conv = layers.BatchNormalization()(conv)
conv = layers.Dropout(drop_rate,seed=42)(conv)

conv = layers.Conv1D(
    channel_2,filter_2,strides=stride,padding='same',activation='relu',
    kernel_initializer=initializers.he_normal(seed=42))(conv)
conv = layers.BatchNormalization()(conv)
conv = layers.Dropout(drop_rate,seed=42)(conv)

conv = layers.Conv1D(
    channel_3,filter_3,strides=stride,padding='same',activation='relu',
    kernel_initializer=initializers.he_normal(seed=42))(conv)
conv = layers.BatchNormalization()(conv)
conv = layers.Dropout(drop_rate,seed=42)(conv)

pool = layers.GlobalAveragePooling1D()(conv)
pool = layers.BatchNormalization()(pool)

outputs = layers.Dense(61,activation='softmax',kernel_initializer=initializers.he_normal(seed=42))(pool)
model = Model(inputs,outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()
# %%
checkpoint_filepath = "./save/simple_cnn_best.hdf5"

save_best = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch', options=None)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(patience = 5,verbose = 1,factor = 0.5) 

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',min_delta=0.0001,
    patience=30,verbose=1)

history = model.fit(
    X_train[:,:,:input_channel],y_train,
    epochs=2000,
    callbacks=[save_best,early_stop,lr_scheduler],
    validation_split=0.2
    )
# %%
model.load_weights(checkpoint_filepath)
model.evaluate(X_test[:,:,:input_channel],y_test)
