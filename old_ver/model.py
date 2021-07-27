# %%
import tensorflow.keras.layers as layers
from tensorflow.keras import regularizers, initializers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
# %%
class CNN_RNN_Resnet:
    def __init__(self,
        leakyrelu_alpha=0.1, input_kernels=20, input_kernel_width=3, input_regularize_coeff=0.1,
        res_kernels=60, res_kernel_width=3, res_num=3, res_regularize_coeff=0.1):
        self.leakyrelu_alpha = leakyrelu_alpha
        self.input_kernels = input_kernels
        self.input_kernel_width = input_kernel_width
        self.input_regularize_coeff = input_regularize_coeff
        self.res_kernels = res_kernels
        self.res_kernel_width = res_kernel_width
        self.res_num = res_num
        self.res_regularize_coeff = res_regularize_coeff
        self._initializer = initializers.he_uniform
        self._regularizer = regularizers.l2
        self._conv = layers.Conv1D
        self._rnn = layers.GRU
        self._normalization = layers.BatchNormalization
        self._activation = layers.LeakyReLU
        self._pool = layers.GlobalAveragePooling1D
        self._dense = layers.Dense
    
    def __call__(self,lr=0.001,seed=42):
        self.seed = seed
        model = self.build_model()
        model.compile(optimizer=optimizers.Adam(learning_rate=lr),
            loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def conv_gru_input_model(self,input_list,input_kernels,input_kernel_width,regularize_coeff,leakyrelu_alpha):
        conv_input = layers.concatenate(input_list)
        conv_1 = self._conv(input_kernels, input_kernel_width*6, padding='same', strides=2,
                            kernel_regularizer=self._regularizer(regularize_coeff),
                            kernel_initializer=self._initializer(seed=self.seed))(conv_input)
        conv_1 = self._normalization()(conv_1)
        conv_1 = self._activation(leakyrelu_alpha)(conv_1)
        

        rnn_1 = self._rnn(input_kernels*2,return_sequences=True,
                        kernel_regularizer=self._regularizer(regularize_coeff),
                        kernel_initializer=self._initializer(seed=self.seed))(conv_1)
        rnn_1 = self._normalization()(rnn_1)
        rnn_1 = self._activation(leakyrelu_alpha)(rnn_1)
        

        conv_2 = self._conv(input_kernels*6, input_kernel_width*3, padding='same', strides=2,
                            kernel_regularizer=self._regularizer(regularize_coeff),
                            kernel_initializer=self._initializer(seed=self.seed))(rnn_1)
        conv_2 = self._normalization()(conv_2)
        conv_2 = self._activation(leakyrelu_alpha)(conv_2)
        

        rnn_2 = self._rnn(input_kernels*6,return_sequences=True,
                        kernel_regularizer=self._regularizer(regularize_coeff),
                        kernel_initializer=self._initializer(seed=self.seed))(conv_2)
        rnn_2 = self._normalization()(rnn_2)
        rnn_2 = self._activation(leakyrelu_alpha)(rnn_2)

        conv_3 = self._conv(input_kernels*6, input_kernel_width*3, padding='same', strides=2,
                            kernel_regularizer=self._regularizer(regularize_coeff),
                            kernel_initializer=self._initializer(seed=self.seed))(rnn_2)
        conv_3 = self._normalization()(conv_3)
        conv_3 = self._activation(leakyrelu_alpha)(conv_3)
        

        rnn_3 = self._rnn(input_kernels*6,return_sequences=True,
                        kernel_regularizer=self._regularizer(regularize_coeff),
                        kernel_initializer=self._initializer(seed=self.seed))(conv_3)
        rnn_3 = self._normalization()(rnn_3)
        rnn_3 = self._activation(leakyrelu_alpha)(rnn_3)

        # conv_4 = self._conv(input_kernels*6, input_kernel_width, padding='same', strides=2,
        #                     kernel_regularizer=self._regularizer(regularize_coeff),
        #                     kernel_initializer=self._initializer(seed=self.seed))(rnn_3)
        # conv_4 = self._normalization()(conv_4)
        # conv_4 = self._activation(leakyrelu_alpha)(conv_4)
        

        # rnn_4 = self._rnn(input_kernels*6,return_sequences=True,
        #                 kernel_regularizer=self._regularizer(regularize_coeff),
        #                 kernel_initializer=self._initializer(seed=self.seed))(conv_4)
        # rnn_4 = self._normalization()(rnn_4)
        # rnn_4 = self._activation(leakyrelu_alpha)(rnn_4)

        conv_model = Model(input_list, rnn_3)
        return conv_model
    
    def conv_block(self,input,kernels,width,regularize_coeff,leakyrelu_alpha):
        conv_1 = self._conv(kernels, width*3,
                    padding='same',
                    strides=1, kernel_regularizer=self._regularizer(regularize_coeff),
                    kernel_initializer=self._initializer(seed=self.seed))(input)
        conv_1 = self._normalization()(conv_1)
        conv_1 = self._activation(leakyrelu_alpha)(conv_1)

        conv_2 = self._conv(kernels*2, width*2,
                    padding='same',
                    strides=1, kernel_regularizer=self._regularizer(regularize_coeff),
                    kernel_initializer=self._initializer(seed=self.seed))(conv_1)
        conv_2 = self._normalization()(conv_2)
        conv_2 = self._activation(leakyrelu_alpha)(conv_2)

        conv_3 = self._conv(kernels, width,
                    padding='same',
                    strides=1, kernel_regularizer=self._regularizer(regularize_coeff),
                    kernel_initializer=self._initializer(seed=self.seed))(conv_2)
        conv_3 = self._normalization()(conv_3)
        conv_3 = self._activation(leakyrelu_alpha)(conv_3)

        return conv_3

    def res_block(self, input, kernels, num_layers, width, regularize_coeff,leakyrelu_alpha):
        shortcut = self._conv(kernels, width*6,
                                        padding='same',
                                        strides=1, kernel_regularizer=self._regularizer(regularize_coeff),
                                        kernel_initializer=self._initializer(seed=self.seed))(input)
        for i in range(num_layers):
            conv = self.conv_block(input,kernels,width,regularize_coeff,leakyrelu_alpha)
            resnet = layers.Add()([conv, shortcut])
            resnet = self._normalization()(resnet)
            shortcut = resnet
            input = resnet
        return resnet

    def build_model(self):

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

        acc_input_diff = layers.Input(shape=(600,3))
        gy_input_diff = layers.Input(shape=(600,3))
        acc_input_diff_mag = layers.Input(shape=(600,1))
        gy_input_diff_mag = layers.Input(shape=(600,1))

        # acc_conv
        acc_input_list = [
            acc_input_xyz, 
            acc_input_mag, 
            acc_input_vol]
        acc_model = self.conv_gru_input_model(acc_input_list,self.input_kernels,self.input_kernel_width,self.input_regularize_coeff,self.leakyrelu_alpha)
        acc_model_2 = self.conv_gru_input_model(acc_input_list,self.input_kernels//2,self.input_kernel_width*2,self.input_regularize_coeff,self.leakyrelu_alpha)

        # gy_conv    
        gy_input_list = [
            gy_input_xyz,
            gy_input_mag,
            gy_input_vol]
        gy_model = self.conv_gru_input_model(gy_input_list,self.input_kernels,self.input_kernel_width,self.input_regularize_coeff,self.leakyrelu_alpha)
        gy_model_2 = self.conv_gru_input_model(gy_input_list,self.input_kernels//2,self.input_kernel_width*2,self.input_regularize_coeff,self.leakyrelu_alpha)

        # xyz_conv
        xyz_input_list = [
            acc_input_xyz,
            gy_input_xyz,
            acc_minus_gy_input,
            acc_cross_gy_input]
        xyz_model = self.conv_gru_input_model(xyz_input_list,self.input_kernels,self.input_kernel_width,self.input_regularize_coeff,self.leakyrelu_alpha)
        xyz_model_2 = self.conv_gru_input_model(xyz_input_list,self.input_kernels//2,self.input_kernel_width*2,self.input_regularize_coeff,self.leakyrelu_alpha)

        # # fft_conv
        fft_input_list = [
            acc_input_fft,
            gy_input_fft,
            acc_input_fft_mag,
            gy_input_fft_mag]
        fft_model = self.conv_gru_input_model(fft_input_list,self.input_kernels,self.input_kernel_width,self.input_regularize_coeff,self.leakyrelu_alpha)
        # fft_model_2 = self.conv_gru_input_model(fft_input_list,input_kernels//2,input_kernel_width*2,leakyrelu_alpha)

        # fft_acc_conv
        fft_acc_input_list = [
            acc_input_fft,
            acc_input_fft_mag]
        fft_acc_model = self.conv_gru_input_model(fft_acc_input_list,self.input_kernels,self.input_kernel_width,self.input_regularize_coeff,self.leakyrelu_alpha)

        # fft_gy_conv
        fft_gy_input_list = [
            gy_input_fft,
            gy_input_fft_mag]
        fft_gy_model = self.conv_gru_input_model(fft_gy_input_list,self.input_kernels,self.input_kernel_width,self.input_regularize_coeff,self.leakyrelu_alpha)

        # diff_comv
        diff_input_list = [
            acc_input_diff,
            gy_input_diff,
            acc_input_diff_mag,
            gy_input_diff_mag
        ]
        diff_model = self.conv_gru_input_model(diff_input_list,self.input_kernels,self.input_kernel_width,self.input_regularize_coeff,self.leakyrelu_alpha)

        # concat layers
        concat = layers.concatenate([
            acc_model.output, 
            gy_model.output,
            fft_model.output, 
            fft_acc_model.output, 
            fft_gy_model.output,
            xyz_model.output,
            diff_model.output,
            acc_model_2.output, 
            gy_model_2.output,
            xyz_model_2.output
            ])

        res_concat = self.res_block(concat, self.res_kernels, self.res_num, self.res_kernel_width, self.res_regularize_coeff, self.leakyrelu_alpha)

        #fc_layers = self._normalization()(res_concat)
        fc_layers = self._pool()(res_concat)
        output_layer = self._dense(61, activation='softmax',
                        kernel_initializer=self._initializer(seed=self.seed))(fc_layers)

        model = Model([
            acc_input_xyz, gy_input_xyz,
            acc_input_mag, gy_input_mag,
            acc_input_vol, gy_input_vol,
            acc_input_fft, gy_input_fft,
            acc_input_fft_mag, gy_input_fft_mag,
            acc_cross_gy_input, acc_minus_gy_input,
            acc_input_diff, gy_input_diff,
            acc_input_diff_mag, gy_input_diff_mag
            ], output_layer)

        # model.compile(optimizer=optimizers.Adam(learning_rate=self.lr),
        #     loss='categorical_crossentropy', metrics=['accuracy'])

        return model     