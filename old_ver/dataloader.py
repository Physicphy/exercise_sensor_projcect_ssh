# %%
import pandas as pd
import numpy as np
import os
import tensorflow.keras as keras
from transforms3d.axangles import axangle2mat
from scipy.fftpack import dct
from sklearn.utils.class_weight import compute_class_weight
# %%
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
train_y = pd.read_csv('./data/data_y_train.csv')  # label load
label_dict = dict()
for label, label_desc in zip(train_y.label, train_y.label_desc):
    label_dict[label] = label_desc
# 'Squat (kettlebell / goblet)'에서 [/]를 [,]으로 변경
label_dict[45] = 'Squat (kettlebell , goblet)'

class_weights = compute_class_weight(
    'balanced',
    np.unique(train_y.label),
    train_y.label)

def class_weight_dict(bias=1):
    return dict(zip(
    list(range(61)),
    class_weights+bias
    ))
# %%
class Workout_dataset(keras.utils.Sequence):
    def __init__(self, load_dir, label_dir,mode ,fold = None,augment=False, batch_size=16, shuffle=True, rot_prob=0.5, perm_prob=0.2):
        self.load_dir = load_dir
        self.idxs = self.get_idxs(fold,mode,label_dir)
        self.label_dict = self.get_label_dict(label_dir)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.augment = augment
        self.on_epoch_end()
        self.rot_prob = rot_prob
        self.perm_prob = perm_prob

    # 현재 모드에 맞게, 선택한 fold에 해당하는 id 리스트를 만든다
    def get_idxs(self, fold, mode,label_dir):
        labels_df = pd.read_csv(label_dir)
        if mode =='Train':
            # mode가 train이면, 지정한 fold가 아닌 것들의 id리스트를 만든다.
            fold_data = labels_df.loc[labels_df.loc[:,'Fold'] != fold, 'id']
            items = fold_data.tolist()
        elif mode =='Valid':
            # mode가 valid이면, 지정한 fold에 해당하는 것들의 id리스트를 만든다.
            fold_data = labels_df.loc[labels_df.loc[:,'Fold'] == fold, 'id']
            items = fold_data.tolist()
        elif mode =='Test':
            # mode가 test이면, 모든 id리스트를 만든다.
            fold_data = labels_df.loc[:, 'id']
            items = fold_data.tolist()
        else :
            # train, valid로 지정을 하지 않으면, load_dir에 있는 *.npy를 읽어오기
            # item[:-4]를 하면 .npy를 떼고, 이름만으로 리스트를 만들 수 있다
            items = os.listdir(self.load_dir)
            items = [item[:-4] for item in items]
        return items

    def get_label_dict(self, label_dir):
        labels_df = pd.read_csv(label_dir)
        label_dict = dict()
        # 각 id의 label이 무엇인지 매칭시켜주는 dict를 만든다.
        for idx in self.idxs:
            label_dict[idx] = labels_df.loc[labels_df['id']==int(idx), 'label']
        return label_dict

    def __len__(self):
        return int(np.floor(len(self.idxs)/self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.idxs))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        # idxs에서, batch_size만큼을 선택
        dummy_indexes = self.indexes[idx *self.batch_size: (idx+1)*self.batch_size]
        batch_indexes = [(self.idxs[i]) for i in dummy_indexes]
        
        # idx.npy를 읽어와서, batch_data를 만들기
        # 총 30개의 feature가 있는 경우 고려.
        batch_data = np.empty((self.batch_size, 600, 30))
        batch_labels = np.zeros((self.batch_size, 61))
        loaded_data = [self.load_npy(i) for i in batch_indexes]

        # batch_labels는 label_dict를 이용해서, one_hot_encoding방식으로
        for num in range(self.batch_size):
            batch_data[num, :, :8] = loaded_data[num]
            batch_labels[num, self.label_dict[batch_indexes[num]]] = 1

        # 데이터 augment를 실행. rotation, rolling
        if self.augment == True:
            batch_data = self.data_augment(batch_data)
        
        # xyz를 모두 곱해서, vol을 계산
        batch_data[:, :,8]=batch_data[:, :,0]*batch_data[:, :,1]*batch_data[:, :,2]
        batch_data[:, :,9]=batch_data[:, :,3]*batch_data[:, :,4]*batch_data[:, :,5]
        
        # 각 데이터를 fourier transform
        for num in range(self.batch_size):
            for m in range(8):
                batch_data[num, :,10+m]=dct(batch_data[num, :,m], type=2, n=600, norm='ortho')
            
        # 같은 축의 acc, gy를 곱하기
        batch_data[:, :,18]=batch_data[:, :,0]*batch_data[:, :,3]
        batch_data[:, :,19]=batch_data[:, :,1]*batch_data[:, :,4]
        batch_data[:, :,20]=batch_data[:, :,2]*batch_data[:, :,5]

        # acc와 gy벡터의 차이 크기
        batch_data[:, :,21]=((batch_data[:, :,0]-batch_data[:, :,3])**2+(batch_data[:, :,1]-batch_data[:, :,4])**2+(batch_data[:, :,2]-batch_data[:, :,5])**2)**0.5

        #xyz와 tot의 변화율
        for num in range(self.batch_size):
            for m in range(8):
                batch_data[num, :,22+m]=np.diff(batch_data[num,:,m], prepend=batch_data[num,:,m][0])

        return [
            batch_data[:, :, :3], batch_data[:, :, 3:6], # acc_xyz, gy_xyz
            batch_data[:, :, 6:7], batch_data[:, :, 7:8], #  acc_mag, gy_mag
            batch_data[:, :, 8:9], batch_data[:, :, 9:10], # acc_vol, gy_vol
            batch_data[:, :, 10:13], batch_data[:, :, 13:16], # ftt_acc_xyz, ftt_gy_xyz
            batch_data[:, :, 16:17], batch_data[:, :, 17:18], # fft_acc_mag, fft_gy_mag
            batch_data[:, :, 18:21], batch_data[:, :, 21:22], # acc_cross_gy, acc_minus_gy
            batch_data[:, :, 22:25], batch_data[:, :, 25:28], # diff_acc_xyz, diff_gy_xyz
            batch_data[:, :, 28:29], batch_data[:, :, 29:30] # diff_acc_mag, diff_gy_mag
            ], batch_labels

    def load_npy(self,idx):
        return np.load(os.path.join(self.load_dir, str(idx)+'.npy'))

    def data_augment(self,data):
        data = self.rolling(data)

        acc_data = data[:,:,0:3]
        gy_data = data[:,:,3:6]

        axis = np.random.uniform(low=-1, high=1, size=3)
        angle = np.random.uniform(low=-np.pi, high=np.pi)

        # 항상 rotation과 permutation 을 시키는게 아니라, 확률적으로 일어나게 설정
        rand_rot = np.random.uniform(low=0,high=1)
        if rand_rot <= self.rot_prob:
            rot_acc_data = self.rotation(acc_data,axis,angle)
            rot_gy_data = self.rotation(gy_data,axis,angle)

            data[:,:,0:3] = rot_acc_data
            data[:,:,3:6] = rot_gy_data

        rand_perm = np.random.uniform(low=0,high=1)
        if rand_perm <= self.perm_prob:
            data = self.permutation(data)
                    
        return data

    def rolling(self,data):
        for j in np.random.choice(data.shape[0], int(data.shape[0]*2/3)):
            data[j] = np.roll(data[j], np.random.choice(data.shape[1]), axis= 0)
        return data

    def rotation(self,data,axis,angle):
        return np.matmul(data , axangle2mat(axis,angle))
    
    def permutation(self,data,n_perm=4,m_sl=10):
        for j in np.random.choice(data.shape[0], int(data.shape[0]*2/3)):
            _data = data[j]
            data_new = np.zeros(_data.shape)
            idx = np.random.permutation(n_perm)
            
            bWhile = True
            while bWhile == True:
                # 600개의 시계열 데이터 내에서 구간을 n_perm개로 나눈다
                # [0, 구간1, 구간2, 구간3, 600]으로 나눌 구간을 만들기
                segs = np.zeros(n_perm+1, dtype=int)
                # 처음과 끝을 제외한 곳에서 세지점을 랜덤으로 선택
                segs[1:-1] = np.sort(np.random.randint(1, (_data.shape[0]-m_sl)//m_sl, n_perm-1))*m_sl
                # 마지막 지점은 데이터 사이즈(600)으로 지정
                segs[-1] = _data.shape[0]
                if np.min(segs[1:]-segs[0:-1]) > m_sl:
                    bWhile = False
            paste_start = 0
            for _i in range(n_perm):
                cut_start = segs[idx[_i]]
                cut_end = segs[idx[_i]+1]
                data_temp = _data[cut_start:cut_end,:]
                paste_end = paste_start+len(data_temp)
                data_new[paste_start:paste_end,:] = data_temp
                paste_start = paste_end
            data[j] = data_new
        return data