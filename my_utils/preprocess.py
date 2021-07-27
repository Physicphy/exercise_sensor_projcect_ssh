# %%
# import os
# os.environ['PYTHONHASHSEED'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
# # terminal에 tf 실행/경고/에러 로그가 출력되지 않게 설정
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# # 재현성위해 세팅
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# # seed 고정
# seed = 42
# import numpy as np
# import random as rn
# np.random.seed(seed)
# rn.seed(seed)
# import tensorflow as tf
# tf.compat.v1.set_random_seed(seed)
# tf.random.set_seed(seed)
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
# %%
class DataPreprocess:
    def __init__(self,folds=10,seed=42,acc_norm=10,gy_norm=2000,features_path='./data/train_features.csv',labels_path='./data/train_labels.csv'):
        self.folds = folds
        self.seed = seed
        self.acc_norm = acc_norm
        self.gy_norm = gy_norm
        self.features_path = features_path
        self.labels_path = labels_path
        self.data_X = pd.read_csv(self.features_path)
        self._add_mag()
        if self.labels_path:
            self.data_y = pd.read_csv(self.labels_path)
            self.labels = self.data_y['label']
            # self._train_test_split()
            # self._add_fold_label()
    
    def _add_mag(self):
        # acc의 총 크기
        self.data_X['acc_mag']=(self.data_X['acc_x']**2+self.data_X['acc_y']**2+self.data_X['acc_z']**2)**0.5

        # gy의 총 크기
        self.data_X['gy_mag']=(self.data_X['gy_x']**2+self.data_X['gy_y']**2+self.data_X['gy_z']**2)**0.5

        # acc와 gy의 크기 조정
        for col in self.data_X.columns:
            if col.startswith('acc'):
                self.data_X[col] = self.data_X[col]/self.acc_norm
            elif col.startswith('gy'):
                self.data_X[col] = self.data_X[col]/self.gy_norm
    
    @property
    def data_list(self):
        data_list = []
        min_idx = self.data_X['id'].min()
        for idx in self.data_X['id'].unique():
            d = np.array(self.data_X.iloc[0+600*(idx-min_idx):600+600*(idx-min_idx),:])
            data_list.append(d)
        return data_list

    def train_test_split(self,test_size=0.2,random_state=42):
        self.X_train, self.X_test, self.y_train_label, self.y_test_label = train_test_split(self.data_list,self.labels, test_size=test_size, random_state=random_state, stratify=self.labels)
        self.y_train = self.data_y.loc[self.y_train_label.index]
        self.y_test = self.data_y.loc[self.y_test_label.index]

    def add_fold_label(self,df_labels):
        df_labels['Fold'] = None
        for label in df_labels.label.unique():
            id_list = df_labels.loc[df_labels.label == label, 'id']
            size = len(id_list)
            fold_list = []
            for i in range(size):
                fold_list.append(i%self.folds)
            np.random.seed(self.seed)
            np.random.shuffle(fold_list)
            for i, _id in enumerate(id_list):
                df_labels.loc[df_labels.id == _id, 'Fold'] = fold_list[i]
        
    def labels_to_csv(self,df_labels,save_path):
        df_labels.to_csv(save_path,index=False)
        print(f"labels_to_csv at {save_path}")

    def features_to_npy(self,data_list,save_path):
        tot_num = len(data_list)
        step = int(tot_num//10)
        for n, d in enumerate(data_list):
            if ((n+1)%step == 0) or n==(tot_num-1):
                print(f"features_to_npy at {save_path} : {n+1}/{tot_num}")
            data_id = int(d[0,0])
            file_name = os.path.join(save_path,str(data_id))
            data_npy = d[:,2:]
            np.save(file_name,data_npy) 