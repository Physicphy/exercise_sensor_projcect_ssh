# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# %%
df_features = pd.read_csv("./data/train_features.csv")
df_features.head()
# %%
df_labels = pd.read_csv("./data/train_labels.csv")
df_labels.head()
# %%
df_features
# %%
df_labels.iloc[:,:-1]
# %%
df_submits = pd.read_csv("./data/test_features.csv")
df_submits.head()
# %%
size_df = pd.DataFrame(df_labels['label'].value_counts(normalize=True))
for i in range(61):
    print(size_df.index[i],float(size_df.iloc[i]))
# %%
size_df[size_df.label < 1/61].shape
# %%
seed = 7
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

from my_utils.preprocess import DataPreprocess
from my_utils.nfold_test import NFoldModel
from my_utils.dataloader import Workout_dataset, class_weight_dict
from my_utils.model import CNN_RNN_Resnet
# %%
class_weight = class_weight_dict(bias=1)
num_of_fold = 10

file_desc = '_bias_1_4step_input_non_diff'

features_path = './data/train_features.csv'
labels_path = './data/train_labels.csv'

preprocess = DataPreprocess(
    folds=num_of_fold,seed=seed,acc_norm=10,gy_norm=2000,
    features_path=features_path,labels_path=labels_path)

preprocess.train_test_split(test_size=0.2,random_state=seed)
preprocess.add_fold_label(preprocess.y_train)
# %%
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
# %%
test_batch_size = 625
if __name__ == '__main__': # (multiprocessing모듈로 gpu메모리 관리)
    predict = nfold_model.nfold_predict(test_batch_size,fold_list)
score_dict = nfold_model.nfold_score(preprocess.y_test_label,predict,average='weighted')
# %%
print("weighted:",score_dict)
score_dict = nfold_model.nfold_score(preprocess.y_test_label,predict,average='macro')
print("macro:",score_dict)
# %%
from sklearn.metrics import log_loss
print("log_loss :",log_loss(preprocess.y_test_label,predict))
# %%
