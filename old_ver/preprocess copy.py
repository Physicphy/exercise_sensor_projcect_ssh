# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
# %%
data_X = pd.read_csv('./data/train_features.csv')
data_y = pd.read_csv('./data/train_labels.csv')
# %%
data_X.shape, data_y.shape
# %%
# for n,col in enumerate(data_X.columns):
#     if col.startswith('acc_'):
#         print("acc",n)
#     elif col.startswith('gy_'):
#         print("gy",n)
# %%
# acc의 총 크기
data_X['acc_tot']=(data_X['acc_x']**2+data_X['acc_y']**2+data_X['acc_z']**2)**0.5

# gy의 총 크기
data_X['gy_tot']=(data_X['gy_x']**2+data_X['gy_y']**2+data_X['gy_z']**2)**0.5

# acc와 gy의 크기 조정
data_X['acc_x'] = data_X['acc_x']/10
data_X['acc_y'] = data_X['acc_y']/10
data_X['acc_z'] = data_X['acc_z']/10
data_X['gy_x'] = data_X['gy_x']/2000
data_X['gy_y'] = data_X['gy_y']/2000
data_X['gy_z'] = data_X['gy_z']/2000

# acc_tot과 gy_tot의 크기 조정
data_X['acc_tot'] = data_X['acc_tot']/10
data_X['gy_tot'] = data_X['gy_tot']/2000
# %%
data_list = []
for idx in data_X['id'].unique():
    d = np.array(data_X.iloc[0+600*idx:600+600*idx,:])
    data_list.append(d)
len(data_list)
# %%
data_list[0].shape
# %%
data_list[0]
# %%
data_X.loc[599]
# %%
labels = data_y['label']
# %%
X_train, X_test, y_train, y_test = train_test_split(data_list,labels, test_size=0.2, random_state=42, stratify=labels)
# %%
np.array(X_train).shape, np.array(X_test).shape, y_train.shape, y_test.shape
# %%
data_y_train = data_y.loc[y_train.index]
# %%
data_y_train['Fold'] = None
for label in data_y_train.label.unique():
    workout_temp_id = data_y_train.loc[data_y_train.label == label, 'id']

    num_of_data = len(workout_temp_id)
    folds = list()
    for i in range(num_of_data):
        folds.append(i%10)

    np.random.seed(42)
    np.random.shuffle(folds)
    for i, id_ in enumerate(workout_temp_id):
        data_y_train.loc[data_y_train.id == id_, 'Fold'] = folds[i]
data_y_train.to_csv('./data/data_y_train.csv',index=False)
# %%
data_y_test = data_y.loc[y_test.index]
data_y_test.to_csv('./data/data_y_test.csv',index=False)
# %%
# 데이터를 npy 파일로 변환 (id별로 파일 하나)
train_save_path = './data/train'
for d in X_train:
    data_id = int(d[0,0])
    file_name = os.path.join(train_save_path,str(data_id))
    data_npy = d[:,2:]
    np.save(file_name,data_npy)

test_save_path = './data/test'
for d in X_test:
    data_id = int(d[0,0])
    file_name = os.path.join(test_save_path,str(data_id))
    data_npy = d[:,2:]
    np.save(file_name,data_npy)

# %%
import pickle
file_name1 = './data/X_test.pickle' # *.p, *.pkl, *.pickle, *.sav 뭘 쓰던 무관!

with open(file_name1, 'wb') as fwb:
    pickle.dump(X_test,fwb)
# %%
import pickle
file_name2 = './data/X_train.pickle' # *.p, *.pkl, *.pickle, *.sav 뭘 쓰던 무관!

with open(file_name2, 'wb') as fwb:
    pickle.dump(X_train,fwb)