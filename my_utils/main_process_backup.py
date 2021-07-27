# %%
import time
start_time = time.time()
print(f">> Start! : {time.ctime(start_time)}")
import os
# terminal에 tf 실행/경고/에러 로그가 출력되지 않게 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# 재현성위해 세팅
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# seed 고정
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
import numpy as np
import tensorflow as tf
import random as rn
tf.random.set_seed(seed)
np.random.seed(seed)
rn.seed(seed)
# 미리 만들어둔 모듈 import
from my_utils.preprocess import DataPreprocess
from my_utils.nfold_test import NFoldModel
from my_utils.dataloader import Workout_dataset
from my_utils.model import CNN_RNN_Resnet
# %%
# data preprocessing
print(">> data preprocessing")
features_path = './data/train_features.csv'
labels_path = './data/train_labels.csv'

num_of_fold = 10

preprocess = DataPreprocess(
    folds=num_of_fold,seed=seed,acc_norm=10,gy_norm=2000,
    features_path=features_path,labels_path=labels_path)

preprocess.train_test_split(test_size=0.2,random_state=42)
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
file_base_name = 'CNN_RNN_Resnet_10_use_mp'
leakyrelu_alpha = 0.1 # 0.1
res_num = 5 # 3, 5, 10
nfold_model = NFoldModel(batch_size=64, valid_ratio=4, lr=0.001, early_stop_patience=30, seed=42)
nfold_model.Model_class = CNN_RNN_Resnet
nfold_model.Workout_dataset = Workout_dataset
nfold_model.model_parameter = dict(leakyrelu_alpha = leakyrelu_alpha, 
    input_kernels = 20, input_kernel_width = 3, input_regularize_coeff=0.001,
    res_kernels = 60, res_kernel_width = 3, res_regularize_coeff=0.1, res_num = res_num
    )
nfold_model.file_base_name = f'{file_base_name}_alpha_{leakyrelu_alpha}_res_num_{res_num}'
fold_list = list(range(num_of_fold))
print(f">>> model_parameter :\n{nfold_model.model_parameter}\n>>> num_of_fold : {num_of_fold}")
# %%
# nfold_train 
print(">> model training")
if __name__ == '__main__': # (multiprocessing모듈로 gpu메모리 관리)
    nfold_model.nfold_train(fold_list)
# %%
# nfold_eval
print(">> evaluate testset")
test_batch_size = 625
if __name__ == '__main__': # (multiprocessing모듈로 gpu메모리 관리)
    nfold_model.nfold_evaluate_test(test_batch_size,fold_list)
# %%
# nfold_predict (avg score)
print(">> calc predict_test and score_test")
if __name__ == '__main__': # (multiprocessing모듈로 gpu메모리 관리)
    predict = nfold_model.nfold_predict(test_batch_size,fold_list)
score_dict = nfold_model.nfold_score(preprocess.y_test_label,predict,average='weighted')
print("weighted:",score_dict)
score_dict = nfold_model.nfold_score(preprocess.y_test_label,predict,average='macro')
print("macro:",score_dict)

end_time = time.time()
print(f">> Finished! : {time.ctime(end_time)}")
t = end_time - start_time
print(f">> whole process spend {round(t//3600)}h {round((t%3600)//60)}m {round(t%60,1)}s")