# %%
import tensorflow.keras.layers as layers
from tensorflow.keras import regularizers, initializers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM
# %%
# import numpy as np
# import tensorflow as tf
# import random as rn
# import os

# # seed 고정
# os.environ['PYTHONHASHSEED'] = str(42)

# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# tf.random.set_seed(42)
# np.random.seed(42)
# rn.seed(42)
# %%
# resnetblock
def conv_block(x,kernels,width,regularize_coeff,leakyrelu_alpha):
    x = layers.Conv1D(kernels, width*3,
                padding='same',
                strides=1, kernel_regularizer=regularizers.l2(regularize_coeff),
                kernel_initializer=initializers.he_uniform(seed=42))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    #x = layers.ELU()(x)

    x = layers.Conv1D(kernels*2, width,
                padding='same',
                strides=1, kernel_regularizer=regularizers.l2(regularize_coeff),
                kernel_initializer=initializers.he_uniform(seed=42))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    #x = layers.ELU()(x)

    x = layers.Conv1D(kernels, width,
                padding='same',
                strides=1, kernel_regularizer=regularizers.l2(regularize_coeff),
                kernel_initializer=initializers.he_uniform(seed=42))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(leakyrelu_alpha)(x)
    #x = layers.ELU()(x)
    return x

def res_block(x, kernels, num_layers, width, regularize_coeff,leakyrelu_alpha=0.1):
    shortcut = layers.Conv1D(kernels, width*3,
                                    padding='same',
                                    strides=1, kernel_regularizer=regularizers.l2(regularize_coeff),
                                    kernel_initializer=initializers.he_uniform(seed=42))(x)
    for i in range(num_layers):
        x = conv_block(x,kernels,width,regularize_coeff,leakyrelu_alpha)
        x = layers.Add()([x, shortcut])
        shortcut = x
    return x
# %%
def conv_gru_input_model(input_list,input_kernels,input_kernel_width,leakyrelu_alpha):
    conv_input = layers.concatenate(input_list)
    conv_1 = layers.Conv1D(input_kernels, input_kernel_width*6, padding='same', strides=2,
                        kernel_initializer=initializers.he_uniform(seed=42))(conv_input)
    conv_1 = layers.LeakyReLU(leakyrelu_alpha)(conv_1)
    #conv_1 = layers.ELU()(conv_1)
    conv_1 = layers.BatchNormalization()(conv_1)

    rnn_1 = layers.GRU(input_kernels*2,return_sequences=True,
                    kernel_initializer=initializers.he_uniform(seed=42))(conv_1)
    rnn_1 = layers.LeakyReLU(leakyrelu_alpha)(rnn_1)
    #rnn_1 = layers.ELU()(rnn_1)
    rnn_1 = layers.BatchNormalization()(rnn_1)

    conv_2 = layers.Conv1D(input_kernels*6, input_kernel_width, padding='same', strides=2,
                        kernel_initializer=initializers.he_uniform(seed=42))(rnn_1)
    conv_2 = layers.LeakyReLU(leakyrelu_alpha)(conv_2)
    #conv_2 = layers.ELU()(conv_2)
    conv_2 = layers.BatchNormalization()(conv_2)

    rnn_2 = layers.GRU(input_kernels*6,return_sequences=True,
                    kernel_initializer=initializers.he_uniform(seed=42))(conv_2)
    rnn_2 = layers.LeakyReLU(leakyrelu_alpha)(rnn_2)
    #rnn_2 = layers.ELU()(rnn_2)
    rnn_2 = layers.BatchNormalization()(rnn_2)

    # conv_3 = layers.Conv1D(input_kernels*3, input_kernel_width, padding='same', strides=2,
    #                     kernel_initializer=initializers.he_uniform(seed=42))(rnn_2)
    # conv_3 = layers.LeakyReLU(leakyrelu_alpha)(conv_3)
    # #conv_3 = layers.ELU()(conv_2)
    # conv_3 = layers.BatchNormalization()(conv_3)

    # rnn_3 = layers.GRU(input_kernels*3,return_sequences=True,activation='tanh')(conv_3)
    # rnn_3 = layers.LeakyReLU(leakyrelu_alpha)(rnn_3)
    # #rnn_3 = layers.ELU()(rnn_3)
    # rnn_3 = layers.BatchNormalization()(rnn_3)

    conv_model = Model(input_list, rnn_2)
    return conv_model
# %%
def make_CNN_RNN_model(
    lr, leakyrelu_alpha,
    input_kernels, input_kernel_width,
    res_kernels, res_kernel_width, res_num, res_regularize_coeff):
    
    acc_input_xyz = layers.Input(shape=(600, 3))
    acc_input_mag = layers.Input(shape=(600, 1))
    acc_input_vol = layers.Input(shape=(600, 1))
    acc_input_fft = layers.Input(shape=(600, 3))
    acc_input_fft_mag = layers.Input(shape=(600, 1))

    gy_input_xyz = layers.Input(shape=(600, 3))
    gy_input_mag = layers.Input(shape=(600, 1))
    gy_input_vol = layers.Input(shape=(600, 1))
    gy_input_fft = layers.Input(shape=(600, 3))
    gy_input_fft_mag = layers.Input(shape=(600, 1))

    acc_cross_gy_input = layers.Input(shape=(600, 3))
    acc_minus_gy_input = layers.Input(shape=(600, 1))
    
    # acc_conv
    acc_input_list = [
        acc_input_xyz, 
        acc_input_mag, 
        acc_input_vol]
    acc_model = conv_gru_input_model(acc_input_list,input_kernels,input_kernel_width,leakyrelu_alpha)
    acc_model_2 = conv_gru_input_model(acc_input_list,input_kernels,input_kernel_width*2,leakyrelu_alpha)
    
    # gy_conv    
    gy_input_list = [
        gy_input_xyz,
        gy_input_mag,
        gy_input_vol]
    gy_model = conv_gru_input_model(gy_input_list,input_kernels,input_kernel_width,leakyrelu_alpha)
    gy_model_2 = conv_gru_input_model(gy_input_list,input_kernels,input_kernel_width*2,leakyrelu_alpha)

    # xyz_conv
    xyz_input_list = [
        acc_input_xyz,
        gy_input_xyz,
        acc_minus_gy_input,
        acc_cross_gy_input]
    xyz_model = conv_gru_input_model(xyz_input_list,input_kernels,input_kernel_width,leakyrelu_alpha)
    xyz_model_2 = conv_gru_input_model(xyz_input_list,input_kernels//2,input_kernel_width*2,leakyrelu_alpha)

    # # mag_vol_conv
    # mag_vol_input_list = [
    #     acc_input_mag,
    #     acc_input_vol,
    #     gy_input_mag,
    #     gy_input_vol,
    #     acc_minus_gy_input]
    # mag_vol_model = conv_gru_input_model(mag_vol_input_list,input_kernels,input_kernel_width,leakyrelu_alpha)

    # # fft_conv
    fft_input_list = [
        acc_input_fft,
        gy_input_fft,
        acc_input_fft_mag,
        gy_input_fft_mag]
    fft_model = conv_gru_input_model(fft_input_list,input_kernels,input_kernel_width,leakyrelu_alpha)
    fft_model_2 = conv_gru_input_model(fft_input_list,input_kernels//2,input_kernel_width*2,leakyrelu_alpha)

    # fft_acc_conv
    fft_acc_input_list = [
        acc_input_fft,
        acc_input_fft_mag]
    fft_acc_model = conv_gru_input_model(fft_acc_input_list,input_kernels,input_kernel_width,leakyrelu_alpha)
    fft_acc_model_2 = conv_gru_input_model(fft_acc_input_list,input_kernels//2,input_kernel_width*2,leakyrelu_alpha)

    # fft_gy_conv
    fft_gy_input_list = [
        gy_input_fft,
        gy_input_fft_mag]
    fft_gy_model = conv_gru_input_model(fft_gy_input_list,input_kernels,input_kernel_width,leakyrelu_alpha)
    fft_gy_model_2 = conv_gru_input_model(fft_gy_input_list,input_kernels//2,input_kernel_width*2,leakyrelu_alpha)

    # # acc_cross_gy_conv
    # acc_gy_comb_input_list = [
    #     acc_cross_gy_input]#, acc_minus_gy_input]
    # acc_gy_comb_model = conv_gru_input_model(acc_gy_comb_input_list,input_kernels,input_kernel_width,leakyrelu_alpha)

    # concat layers
    concat = layers.concatenate([
        acc_model.output, 
        gy_model.output,
        # mag_vol_model.output, 
        # acc_gy_comb_model.output,
        fft_model.output, 
        fft_acc_model.output, 
        fft_gy_model.output,
        xyz_model.output,
        acc_model_2.output, 
        gy_model_2.output,
        # fft_model_2.output, 
        # fft_acc_model_2.output, 
        # fft_gy_model_2.output,
        xyz_model_2.output
        ])
    
    res_concat = res_block(concat, res_kernels, res_num, res_kernel_width, res_regularize_coeff)

    # rnn = layers.GRU(60,return_sequences=True)(res_concat)
    # rnn = layers.LeakyReLU(leakyrelu_alpha)(rnn)
    # rnn = layers.BatchNormalization()(rnn)
    # rnn = layers.GRU(60)(rnn)
    # rnn = layers.LeakyReLU(leakyrelu_alpha)(rnn)
    # rnn = layers.BatchNormalization()(rnn)
    # fc_layers = layers.Dense(61, activation='softmax')(rnn)

    fc_layers = layers.BatchNormalization()(res_concat)
    fc_layers = layers.GlobalAveragePooling1D()(fc_layers)
    fc_layers = layers.Dense(61, activation='softmax',
                    kernel_initializer=initializers.he_uniform(seed=42))(fc_layers)

    model = Model([
        acc_input_xyz, gy_input_xyz,
        acc_input_mag, gy_input_mag,
        acc_input_vol, gy_input_vol,
        acc_input_fft, gy_input_fft,
        acc_input_fft_mag, gy_input_fft_mag,
        acc_cross_gy_input, acc_minus_gy_input
        ], fc_layers)

    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                loss='categorical_crossentropy', metrics=['accuracy'])
    return model