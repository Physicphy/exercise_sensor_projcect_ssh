# %%
import time
start_time = time.time()
print(f">> Start! : {time.ctime(start_time)}")

# seed 고정
seed = 42 # 42, 7
print(f">>> seed = {seed}")
import os
os.environ['PYTHONHASHSEED'] = str(seed) #'0'
# GPU가 여럿일 때 특정 GPU를 선택
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
# terminal에 tf 실행/경고/에러 로그가 출력되지 않게 설정
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# tensorflow 재현성위해 세팅
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import tensorflow as tf
# tf.compat.v1.set_random_seed(seed)
tf.random.set_seed(seed)

import numpy as np
import random as rn
np.random.seed(seed)
rn.seed(seed)


# from tensorflow.keras.backend import manual_variable_initialization 
# manual_variable_initialization(True)
# 미리 만들어둔 모듈 import
from my_utils.preprocess import DataPreprocess
from my_utils.nfold_test import NFoldModel
from my_utils.dataloader import Workout_dataset, class_weight_dict
from my_utils.model import CNN_RNN_Resnet #, nonBN_CNN_RNN_Resnet
# %%
# data preprocessing
print(">> data preprocessing")
features_path = './data/train_features.csv'
labels_path = './data/train_labels.csv'

num_of_fold = 5 # 5, 10
# %%
preprocess = DataPreprocess(
    folds=num_of_fold,seed=seed,acc_norm=10,gy_norm=2000,
    features_path=features_path,labels_path=labels_path)

preprocess.train_test_split(test_size=0.2,random_state=seed)
preprocess.add_fold_label(preprocess.y_train)
# %%
# X_train, X_test, y_train, y_test를 npy, csv로 저장
print(">> X_train, X_test, y_train, y_test --(save)--> *.npy, *.csv")
preprocess.labels_to_csv(preprocess.y_train,'./data/y_train.csv')
preprocess.labels_to_csv(preprocess.y_test,'./data/y_test.csv')
preprocess.features_to_npy(preprocess.X_train,"./data/X_train")
preprocess.features_to_npy(preprocess.X_test,"./data/X_test")
# %%
# 모델 구성
print(">> model building")

rot_prob = 0.5 # 0.5
perm_prob = 0.5 # 0.2, 0.5
class_weight = class_weight_dict(bias=1) # None
file_desc = f'_bias_1_seed{seed}' #'_bias_1_ir_coef_1E-3'# '_bias_5e-1_ir_coef_1E-2'#''

nfold_model = NFoldModel(
    batch_size=64, valid_ratio=4, 
    early_stop_patience=30, seed=seed, 
    optimizer='Adam', #'SGD'
    optimizer_setting={'learning_rate':0.001}# {'learning_rate':0.03,'momentum':0.9}  # {'learning_rate':0.2,'momentum':0.9}
    )
nfold_model.Model_class = CNN_RNN_Resnet # nonBN_CNN_RNN_Resnet #
nfold_model.Workout_dataset = Workout_dataset
nfold_model.model_parameter = dict(
    leakyrelu_alpha = 0.1, # 0.1
    input_kernels = 20, 
    input_kernel_width = 3, 
    input_regularize_coeff=0.001, # 0.001, 0.0001, 0.01
    res_kernels = 60, 
    res_kernel_width = 3, 
    res_regularize_coeff=0.1, # 0.1, 0.01
    res_num = 5 # 3, 5, 10
    )
# fold_list = [1,5,9]
fold_list = list(range(num_of_fold))
model_name = nfold_model.Model_class.__name__
add_name = '_and_class_weight' if class_weight else ''
file_base_name = f'{model_name}_{num_of_fold}_use_mp{add_name+file_desc}'
nfold_model.file_base_name = f"{file_base_name}_alpha_{'%.0E' % nfold_model.model_parameter['leakyrelu_alpha']}_res_num_{nfold_model.model_parameter['res_num']}_opt_{nfold_model.optimizer}"
print(f">>> file_base_name : {nfold_model.file_base_name}")
print(f">>> model : {model_name}")
print(f">>> model_parameter :\n{nfold_model.model_parameter}\n>>> num_of_fold : {num_of_fold}\n>>> optimizer : {nfold_model.optimizer}",nfold_model.optimizer_setting)
print(f">>> rot_prob: {rot_prob}, perm_prob: {perm_prob}")
# %%
# nfold_train 
print(">> model training")
if __name__ == '__main__': # (multiprocessing모듈로 gpu메모리 관리)
    nfold_model.nfold_train(
        fold_list,
        class_weight_dict=class_weight,
        rot_prob=rot_prob,
        perm_prob=perm_prob
        )
# %%
# nfold_eval
# print(">> evaluate testset")
# test_batch_size = 625
# if __name__ == '__main__': # (multiprocessing모듈로 gpu메모리 관리)
#     nfold_model.nfold_evaluate_test(test_batch_size,fold_list)
# %%
# nfold_predict (avg score)
print(">> calc predict_test and score_test")
test_batch_size = 625
if __name__ == '__main__': # (multiprocessing모듈로 gpu메모리 관리)
    predict = nfold_model.nfold_predict(test_batch_size,fold_list)
score_dict = nfold_model.nfold_score(preprocess.y_test_label,predict,average='weighted')
print("weighted:",score_dict)
score_dict = nfold_model.nfold_score(preprocess.y_test_label,predict,average='macro')
print("macro:",score_dict)

from sklearn.metrics import log_loss
print("log_loss :",log_loss(preprocess.y_test_label,predict))

print(">> save prediction of testset as pickle")
import pickle
file_name = 'prediction_of_testset.pickle'
with open(file_name, 'wb') as fwb:
    pickle.dump(predict,fwb)

end_time = time.time()
print(f">> Finished! : {time.ctime(end_time)}")
t = end_time - start_time
print(f">> whole process spend {round(t//3600)}h {round((t%3600)//60)}m {round(t%60,1)}s")
# %%
print("\n"+"---"*10+"\n")
# %%
print(">> Start submit.py")
# %%
preprocess_submit = DataPreprocess(
    features_path = "./data/test_features.csv",
    labels_path=None)
# %%
preprocess_submit.features_to_npy(preprocess_submit.data_list,"./data/X_submit")
# %%
submit_batch_size = len(preprocess_submit.data_list)
print("submit data size :",submit_batch_size)
# %%
save_file_name = f"submit_{nfold_model.file_base_name}"
print(f">>> svae file name {save_file_name}")
nfold_model_submit = NFoldModel(optimizer=nfold_model.optimizer,optimizer_setting=nfold_model.optimizer_setting)
nfold_model_submit.Model_class = nfold_model.Model_class
nfold_model_submit.Workout_dataset = nfold_model.Workout_dataset
nfold_model_submit.model_parameter = nfold_model.model_parameter
# fold_list = list(range(num_of_fold))
nfold_model_submit.file_base_name = nfold_model.file_base_name
#nfold_model_submit.file_base_name = f'{file_base_name}'
print(">> submit dataset predict")
if __name__ == '__main__': # (multiprocessing모듈로 gpu메모리 관리)
    predict_submit = nfold_model_submit.nfold_predict(submit_batch_size,fold_list,mode='Submit')
# predict_submit = nfold_model.nfold_predict2(test_batch_size,fold_list,mode='Submit')
# %%
predict_submit.shape
# %%
import pandas as pd
submit_form = pd.read_csv("./data/sample_submission.csv")
# %%
submit_form.iloc[:,1:] = predict_submit
# %%
submit_form.to_csv(f'./data/y_submit/{save_file_name}.csv',index=False)
print(">> Finished!")
# %%
print(">> model_summary")
nfold_model.make_model().summary()