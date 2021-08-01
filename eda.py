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
seed = 42
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
num_of_fold = 5

file_desc = f'_bias_1_seed{seed}'

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
import pickle
file_name = 'prediction_of_testset.pickle'

with open(file_name, 'rb') as frb:
    loaded_predict = pickle.load(frb)
# %%
predict_label = np.argmax(loaded_predict,axis=1)
# %%
true_label = np.array(preprocess.y_test_label)
# %%
true_label.shape, predict_label.shape
# %%
o_dict = {}
x_dict = {}
n_dict = {}
for i in range(len(true_label)):
    t = true_label[i]
    p = predict_label[i]
    n_dict[t] = n_dict.get(t,0)+1
    if t == p:
        o_dict[t] = o_dict.get(t,0)+1
    else:
        x_dict[t] = x_dict.get(t,[])+[p]
# %%
o_dict
# %%
n_dict
# %%
x_dict
# %%
df_result = pd.DataFrame.from_dict(n_dict,orient='index',columns=['tot_num'])
df_result
# %%
df_result.loc[:,'correct_predict'] = 0
df_result.loc[:,'wrong_predict'] = None
df_result
# %%
for i in df_result.index:
    df_result.loc[i,'correct_predict'] = o_dict.get(i,0)
    df_result.loc[i,'wrong_predict'] = str(x_dict.get(i,None))
df_result
# %%
df_result['ratio'] = df_result['correct_predict']/df_result['tot_num']
df_result
# %%
# 하나도 못 맞춘 것
set(n_dict) - set(o_dict)
# %%
# 전부 맞춘 것
set(n_dict) - set(x_dict)
# %%
x_dict[0]
# %%
x_dict[12]
# %%
x_dict[32]
# %%
x_dict[49]
# %%
x_list = []
for k in x_dict:
    x_list = x_list + x_dict[k]
len(x_list)
# %%
73/625
# %%
x_list.count(26), len(x_list), x_list.count(26)/len(x_list)
# %%
from collections import Counter
Counter(x_list)
# %%
ratio_dict = {}
for k in n_dict:
    ratio_dict[k] = (o_dict.get(k,0)/n_dict[k])
ratio_dict
# %%
ratio_dict_sorted = dict(sorted(ratio_dict.items(),key=(lambda x:x[1])))
# %%
sns.set(rc = {'figure.figsize':(15,8)})
sns.barplot(data=df_result,x=df_result.index,y='ratio',order=df_result.sort_values(by='ratio').index)
# %%
sns.barplot(data=df_result.sort_values(by='ratio').iloc[:10],x=df_result.sort_values(by='ratio').iloc[:10].index,y='ratio',order=df_result.sort_values(by='ratio').iloc[:10].index)
# %%
df_result.loc[51]
# %%
df_result.sort_values(by='tot_num',ascending=False)['tot_num']/df_result['tot_num'].sum()
# %%
wrong_predict = Counter(x_list)
# %%
key = list(wrong_predict.keys())
# %%
ratio = list(wrong_predict.values())
ratio = [ round(r/sum(ratio),4) for r in ratio]
ratio
# %%
plt.pie(ratio, labels=key, autopct='%.2f%%')
plt.show()
# %%
wrong_predict
# %%
wrong_predict_reshape = {}
for k in wrong_predict:
    if wrong_predict[k] > 3:
        wrong_predict_reshape[k] = wrong_predict[k]
    else:
        wrong_predict_reshape['other'] = wrong_predict_reshape.get('other',0)+wrong_predict[k]
wrong_predict_reshape
# %%
key = [26,60,48,30,'otehr']
ratio = [round(r/73,4) for r in [18,8,4,4,39]]
ratio
# %%
plt.pie(ratio, labels=key, autopct='%.2f%%')
plt.show()
# %%
df_result[df_result.ratio>=0.8].mean()
# %%
