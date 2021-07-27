# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
from my_utils import Workout_dataset

import os

os.environ['PYTHONHASHSEED'] = str(42)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

tf.random.set_seed(42)
np.random.seed(42)
rn.seed(42)
# %%
train_dir = './data/train'
label_dir = './data/data_y_train.csv'

def scheduler(epoch, lr):
    if (epoch>20) and (lr > 0.00001):
        lr = lr*0.9
        return lr
    else:
        return lr

train_y = pd.read_csv('./data/data_y_train.csv')  # label load
label_dict = dict()
for label, label_desc in zip(train_y.label, train_y.label_desc):
    label_dict[label] = label_desc
# 'Squat (kettlebell / goblet)'에서 [/]를 [,]으로 변경
label_dict[45] = 'Squat (kettlebell , goblet)'
# %%
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
BATCH_SIZE = 64

train_loader = Workout_dataset(
    train_dir, label_dir, mode='Train',
    fold=0, batch_size=BATCH_SIZE, augment=True, shuffle=True)

valid_loader = Workout_dataset(
    train_dir, label_dir, mode='Valid',
    fold=0, batch_size=16, shuffle=True)

# %%
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    'balanced',
    np.unique(train_y.label),
    train_y.label)

class_weight_dict = dict(zip(
    list(range(61)),
    class_weights+1
    ))
# %%
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, GlobalAveragePooling1D, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import he_normal
# %%
kernel_size1 = 19
kernel_size2 = 3
kernel_size3 = 5
kk = 10
stride = 2

acc_input_1 = layers.Input(shape=(600, 3)) # xyz
acc_input_2 = layers.Input(shape=(600, 1)) # tot mag
acc_input_3 = layers.Input(shape=(600, 1)) # vol
acc_input_4 = layers.Input(shape=(600, 3)) # ftt_xyz
gy_input_1 = layers.Input(shape=(600, 3)) 
gy_input_2 = layers.Input(shape=(600, 1)) 
gy_input_3 = layers.Input(shape=(600, 1))
gy_input_4 = layers.Input(shape=(600, 3))



# conv1_inputs = layers.concatenate([acc_input_1, acc_input_2, gy_input_1, gy_input_2])
conv1_inputs = layers.concatenate([acc_input_1, acc_input_2, acc_input_3])

conv1 = Conv1D(
    kk,kernel_size1,strides=stride,padding='same',activation='elu',
    kernel_initializer=he_normal(seed=42)
    )(conv1_inputs)
conv1 = BatchNormalization()(conv1)
#conv1 = Dropout(0.2,seed=42)(conv1)
gru1 = layers.GRU(kk*2,return_sequences=True)(conv1)
gru1 = layers.LeakyReLU(0.2)(gru1)
gru1 = BatchNormalization()(gru1)
conv1 = Conv1D(
    kk*6,kernel_size2,strides=stride,padding='same',activation='elu',
    kernel_initializer=he_normal(seed=42)
    )(gru1)
conv1 = BatchNormalization()(conv1)
#conv1 = Dropout(0.2,seed=42)(conv1)
gru1 = layers.GRU(kk*6,return_sequences=True)(conv1)
gru1 = layers.LeakyReLU(0.2)(gru1)
gru1 = BatchNormalization()(gru1)

acc_model = Model([acc_input_1, acc_input_2, acc_input_3], gru1)


conv2_inputs = layers.concatenate([gy_input_1, gy_input_2, gy_input_3])

conv2 = Conv1D(
    kk,kernel_size1,strides=stride,padding='same',activation='elu',
    kernel_initializer=he_normal(seed=42)
    )(conv2_inputs)
conv2 = BatchNormalization()(conv2)
#conv2 = Dropout(0.1,seed=42)(conv2)
gru2 = layers.GRU(kk*2,return_sequences=True)(conv2)
gru2 = layers.LeakyReLU(0.2)(gru2)
gru2 = BatchNormalization()(gru2)
conv2 = Conv1D(
    kk*6,kernel_size2,strides=stride,padding='same',activation='elu',
    kernel_initializer=he_normal(seed=42)
    )(gru2)
conv2 = BatchNormalization()(conv2)
#conv2 = Dropout(0.1,seed=42)(conv2)
gru2 = layers.GRU(kk*6,return_sequences=True)(conv2)
gru2 = layers.LeakyReLU(0.2)(gru2)
gru2 = BatchNormalization()(gru2)

gy_model = Model([gy_input_1, gy_input_2, gy_input_3], gru2)

conv3_inputs = layers.concatenate([acc_input_2,gy_input_2, acc_input_3, gy_input_3])

conv3 = Conv1D(
    kk,kernel_size1,strides=stride,padding='same',activation='elu',
    kernel_initializer=he_normal(seed=42)
    )(conv3_inputs)
conv3 = BatchNormalization()(conv3)
#conv3 = Dropout(0.1,seed=42)(conv3)
gru3 = layers.GRU(kk*2,return_sequences=True)(conv3)
gru3 = layers.LeakyReLU(0.2)(gru3)
gru3 = BatchNormalization()(gru3)
conv3 = Conv1D(
    kk*6,kernel_size2,strides=stride,padding='same',activation='elu',
    kernel_initializer=he_normal(seed=42)
    )(gru3)
conv3 = BatchNormalization()(conv3)
#conv3 = Dropout(0.1,seed=42)(conv3)
gru3 = layers.GRU(kk*6,return_sequences=True)(conv3)
gru3 = layers.LeakyReLU(0.2)(gru3)
gru3 = BatchNormalization()(gru3)

mag_model = Model([acc_input_2,gy_input_2, acc_input_3, gy_input_3], gru3)

conv4_inputs = layers.concatenate([acc_input_4, gy_input_4])

conv4 = Conv1D(
    kk,kernel_size1,strides=stride,padding='same',activation='elu',
    kernel_initializer=he_normal(seed=42)
    )(conv4_inputs)
conv4 = BatchNormalization()(conv4)
#conv4 = Dropout(0.1,seed=42)(conv4)
gru4 = layers.GRU(kk*2,return_sequences=True)(conv4)
gru4 = layers.LeakyReLU(0.2)(gru4)
gru4 = BatchNormalization()(gru4)
conv4 = Conv1D(
    kk*6,kernel_size2,strides=stride,padding='same',activation='elu',
    kernel_initializer=he_normal(seed=42)
    )(gru4)
conv4 = BatchNormalization()(conv4)
#conv4 = Dropout(0.1,seed=42)(conv4)
gru4 = layers.GRU(kk*6,return_sequences=True)(conv4)
gru4 = layers.LeakyReLU(0.2)(gru4)
gru4 = BatchNormalization()(gru4)

fft_xyz_model = Model([acc_input_4, gy_input_4], gru4)

conv5_inputs = layers.concatenate([acc_input_1, gy_input_1])

conv5 = Conv1D(
    kk,kernel_size1,strides=stride,padding='same',activation='elu',
    kernel_initializer=he_normal(seed=42)
    )(conv5_inputs)
conv5 = BatchNormalization()(conv5)
#conv4 = Dropout(0.1,seed=42)(conv4)
gru5 = layers.GRU(kk*2,return_sequences=True)(conv5)
gru5 = layers.LeakyReLU(0.2)(gru5)
gru5 = BatchNormalization()(gru5)
conv5 = Conv1D(
    kk*6,kernel_size2,strides=stride,padding='same',activation='elu',
    kernel_initializer=he_normal(seed=42)
    )(gru5)
conv5 = BatchNormalization()(conv5)
#conv4 = Dropout(0.1,seed=42)(conv4)
gru5 = layers.GRU(kk*6,return_sequences=True)(conv5)
gru5 = layers.LeakyReLU(0.2)(gru5)
gru5 = BatchNormalization()(gru5)

xyz_model = Model([acc_input_1, gy_input_1], gru5)


concat = layers.concatenate([
    acc_model.output,gy_model.output,
    mag_model.output,xyz_model.output,
    fft_xyz_model.output,
    ])
    
conv_tot = Conv1D(
    kk*6,kernel_size3,strides=stride,padding='same',activation='elu',
    kernel_initializer=he_normal(seed=42)
    )(concat)
conv_tot = BatchNormalization()(conv_tot)

# pool = layers.AveragePooling1D(9,strides=3,padding='same')(conv1)

gru = layers.GRU(60)(conv_tot)
gru = layers.LeakyReLU(0.2)(gru)
gru = BatchNormalization()(gru)
#gru = Dropout(0.2,seed=42)(gru)
# gru = layers.GRU(120,return_sequences=True)(concat)
# gru = layers.LeakyReLU(0.2)(gru)
# gru = BatchNormalization()(gru)
# gru = Dropout(0.5,seed=42)(gru)
# gru = layers.GRU(60)(gru)
# gru = layers.LeakyReLU(0.2)(gru)
# gru = BatchNormalization()(gru)
# gru = Dropout(0.2,seed=42)(gru)
outputs = Dense(61, activation='softmax')(gru)

# pool = GlobalAveragePooling1D()(conv_tot)
# outputs = Dense(61, activation='softmax')(pool)

model = Model([acc_input_1,gy_input_1,acc_input_2,gy_input_2,acc_input_3,gy_input_3,acc_input_4,gy_input_4],outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
# %%
checkpoint_filepath = "./save/simple_cnn_best.hdf5"

save_best = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch', options=None)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',min_delta=0.0001,
    patience=50,verbose=1)

history = model.fit_generator(
    generator=train_loader,
    validation_data=valid_loader,
    epochs=2000,
    callbacks=[save_best,early_stop,lr_scheduler],
    class_weight=class_weight_dict)
# %%
test_dir = './data/test'
test_label_dir = './data/data_y_test.csv'
# %%
test_loader = Workout_dataset(
        test_dir, test_label_dir, mode='Test', batch_size=625, shuffle=False)
# %%
model.load_weights(checkpoint_filepath)
model.evaluate_generator(generator=test_loader,verbose=1)
# %%
