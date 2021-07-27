# %%
import os
# terminal에 tf 실행/경고/에러 로그가 출력되지 않게 설정
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
os.environ['TF_CUDNN_USE_AUTOTUNE'] ='0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# 재현성위해 세팅
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# seed 고정
seed = 42
import numpy as np
import random as rn
rn.seed(seed)
np.random.seed(seed)

import tensorflow as tf
tf.compat.v1.set_random_seed(seed)
tf.random.set_seed(seed)
from tensorflow.keras.backend import manual_variable_initialization 
manual_variable_initialization(True)
# from tensorflow.keras import backend as K
# session_conf = tf.compat.v1.ConfigProto(
#     intra_op_parallelism_threads=1,
#     inter_op_parallelism_threads=1)

# #Force Tensorflow to use a single thread
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

# tf.compat.v1.keras.backend.set_session(sess)
# K.set_learning_phase(0)

from my_utils.preprocess import DataPreprocess
preprocess = DataPreprocess(
    features_path = "./data/test_features.csv",
    labels_path=None)
# %%
preprocess.features_to_npy(preprocess.data_list,"./data/X_submit")
# %%
test_batch_size = len(preprocess.data_list)
print("submit data size :",test_batch_size)
# %%
from my_utils.nfold_test import NFoldModel

num_of_fold = 10
file_base_name = 'CNN_RNN_Resnet_10_use_mp'
leakyrelu_alpha = 0.1 # -0.1, 0.1
res_num = 5 # 3, 5, 10
optimizer = 'SGD'
optimizer_setting={'momentum':0.5}
save_file_name = f'submit_alpha_{leakyrelu_alpha}_res_num_{res_num}_opt_{optimizer}'
#save_file_name = f'submit_test_batchsize64'
nfold_model = NFoldModel(optimizer=optimizer,optimizer_setting=optimizer_setting)
nfold_model.model_parameter = dict(leakyrelu_alpha = leakyrelu_alpha, 
    input_kernels = 20, input_kernel_width = 3, input_regularize_coeff=0.001,
    res_kernels = 60, res_kernel_width = 3, res_regularize_coeff=0.1, res_num = res_num
    )
fold_list = [1,5,9]
# fold_list = list(range(num_of_fold))
nfold_model.file_base_name = f'{file_base_name}_alpha_{leakyrelu_alpha}_res_num_{res_num}_opt_{optimizer}'
#nfold_model.file_base_name = f'{file_base_name}'
print(">> submit dataset predict")
if __name__ == '__main__': # (multiprocessing모듈로 gpu메모리 관리)
    predict = nfold_model.nfold_predict(test_batch_size,fold_list,mode='Submit')
# predict = nfold_model.nfold_predict2(test_batch_size,fold_list,mode='Submit')
# %%
predict.shape
# %%
import pandas as pd
submit_form = pd.read_csv("./data/sample_submission.csv")
# %%
submit_form.iloc[:,1:] = predict
# %%
submit_form.to_csv(f'./data/{save_file_name}.csv',index=False)
print(">> Finished!")