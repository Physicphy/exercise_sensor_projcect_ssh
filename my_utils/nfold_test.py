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
from my_utils.dataloader import Workout_dataset
from my_utils.model import CNN_RNN_Resnet
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import time
import multiprocessing as mp
import tensorflow as tf
# %%
class NFoldModel:
    def __init__(self,batch_size=64,valid_ratio=4,early_stop_patience=30,seed=42,checkpoint_name='checkpoint',optimizer='SGD',optimizer_setting={'learning_rate':0.001,'momentum':0.5}):
        self.Model_class = CNN_RNN_Resnet
        self.Workout_dataset = Workout_dataset

        self.train_dir = './data/X_train'
        self.label_dir = './data/y_train.csv'
        self.test_dir = './data/X_test'
        self.test_label_dir = './data/y_test.csv'
        self.submit_dir = "./data/X_submit"
        self.checkpoint_filepath = './save/'
        self.file_base_name = checkpoint_name

        self.batch_size = batch_size
        self.valid_ratio = valid_ratio
        self.valid_batch_size = self.batch_size//self.valid_ratio
        self.seed = seed
        self.optimizer = optimizer
        self.optimizer_setting = optimizer_setting
        self.early_stop_patience = early_stop_patience
        self.model_parameter = dict(leakyrelu_alpha = 0.1,
            input_kernels = 20, input_kernel_width = 3, input_regularize_coeff=0.001,
            res_kernels = 60, res_kernel_width = 3, res_regularize_coeff=0.1, res_num = 5)
        
    def dataset(self,mode,select_fold=None,batch_size=64,shuffle=False,augment=False,rot_prob=0.5,perm_prob=0.2):
        if mode in['Train','Valid']:
            feature_dir = self.train_dir
            label_dir = self.label_dir
        elif mode == 'Test':
            feature_dir = self.test_dir
            label_dir = self.test_label_dir
        else:
            feature_dir = self.submit_dir
            label_dir = self.label_dir

        return self.Workout_dataset(
            feature_dir, label_dir, mode=mode,
            fold=select_fold, batch_size=batch_size, shuffle=shuffle,
            augment=augment, rot_prob=rot_prob,perm_prob=perm_prob)

    def lr_scheduler(self,patience=4,factor=0.5):
        return ReduceLROnPlateau(patience = patience, verbose = 1, factor = factor)
    
    def checkpoint(self,file_name,save_best_only=True,save_weights_only=True):
        path = self.checkpoint_filepath+file_name
        return ModelCheckpoint(
            filepath=path, monitor='val_loss', verbose=1, save_best_only=save_best_only,
            save_weights_only=save_weights_only, mode='auto', save_freq='epoch', options=None)
    
    def early_stop(self):
        return EarlyStopping(monitor='val_loss',min_delta=0.0001,
            patience=self.early_stop_patience,verbose=1)

    def make_model(self):
        model = self.Model_class(**self.model_parameter)(seed=self.seed,optimizer=self.optimizer,optimizer_setting=self.optimizer_setting)
        return model
    
    def train(self,fold,class_weight_dict,rot_prob=0.5,perm_prob=0.2):
        file_name = f'{self.file_base_name}_fold_{fold}.hdf5'
        checkpoint = self.checkpoint(file_name)
        early_stop = self.early_stop()
        lr_scheduler = self.lr_scheduler()
        train_ds = self.dataset('Train',fold,self.batch_size,shuffle=True,augment=True,rot_prob=rot_prob,perm_prob=perm_prob)
        valid_ds = self.dataset('Valid',fold,self.valid_batch_size,shuffle=True)
        model = self.make_model()
        model.fit(train_ds,
            validation_data=valid_ds,
            epochs=1000, verbose=2,
            callbacks=[checkpoint,early_stop,lr_scheduler],
            class_weight=class_weight_dict)

    def nfold_train(self,fold_list,class_weight_dict,rot_prob=0.5,perm_prob=0.2):
        mp_train = mp_run(self.train,print_spend_time=True)
        for fold in fold_list:
            mp_train(fold,class_weight_dict,rot_prob=rot_prob,perm_prob=perm_prob)
            time.sleep(4)
    
    def evaluate_test(self,test_ds,fold):
        checkpoint = f'{self.checkpoint_filepath}{self.file_base_name}_fold_{fold}.hdf5'
        model = self.make_model()
        model.load_weights(checkpoint)
        # model = tf.keras.models.load_model(checkpoint)
        model.evaluate(test_ds,verbose=1)
    
    def nfold_evaluate_test(self,test_batch_size,fold_list):
        test_ds = self.dataset(mode='Test',batch_size=test_batch_size)
        # model = self.make_model()
        mp_eval_test = mp_run(self.evaluate_test)
        for fold in fold_list:
            mp_eval_test(test_ds,fold)
    
    def predict_test(self,test_ds,fold):
        # tf.compat.v1.global_variables_initializer()
        checkpoint = f'{self.checkpoint_filepath}{self.file_base_name}_fold_{fold}.hdf5'
        model = self.make_model()
        model.load_weights(checkpoint)
        # model = tf.keras.models.load_model(checkpoint)
        predict = model.predict(test_ds)
        return predict
    
    def nfold_predict(self,test_batch_size,fold_list,mode='Test'):
        predict = np.zeros((test_batch_size, 61))
        test_ds = self.dataset(mode=mode,batch_size=test_batch_size)
        # model = self.make_model()
        mp_pred_test = mp_run(self.predict_test)
        for fold in fold_list:
            p = mp_pred_test(test_ds,fold)
            predict = predict + p
            time.sleep(4)
        predict = predict/len(fold_list)
        return predict

    def predict_test2(self,model,test_ds,fold):
        checkpoint = f'{self.checkpoint_filepath}{self.file_base_name}_fold_{fold}.hdf5'
        model.load_weights(checkpoint)
        predict = model.predict(test_ds)
        return predict

    def nfold_predict2(self,test_batch_size,fold_list,mode='Test'):
        predict = np.zeros((test_batch_size, 61))
        test_ds = self.dataset(mode=mode,batch_size=test_batch_size)
        model = self.make_model()
        for fold in fold_list:
            p = self.predict_test2(model,test_ds,fold)
            predict = predict + p
        predict = predict/len(fold_list)
        return predict
    
    def nfold_score(self,y_true,predict,average='weighted'):
        score_dict = {}
        predict_labels = np.argmax(predict,axis=1)
        score_dict['accuracy'] = accuracy_score(y_true,predict_labels)
        score_dict['recall'] = recall_score(y_true,predict_labels,average=average)
        score_dict['precision'] = precision_score(y_true,predict_labels,average=average)
        score_dict['f1'] = f1_score(y_true,predict_labels,average=average)
        return score_dict
# %%
# multiprocessing을 이용해서, train을 여러차례 하게 되면,
# 한회가 끝날 때마다 gpu메모리를 해제하고 이어갈 수 있게 한다.
# new_func = mp_run(old_func)로 함수를 새로 만들고,
# new_func(*args,**kwargs)같이 새로만든 함수를 사용
class mp_run:
    def __init__(self,func,print_spend_time=False):
        self.func = func
        self.print_spend_time = print_spend_time

    def __call__(self,*args,**kwargs):
        return self.process(self.func,*args,**kwargs)

    def wrapper_func(self,func,queue,*args,**kwargs):
        result = func(*args,**kwargs)
        queue.put(result)

    def process(self,func,*args,**kwargs):
        # mp.set_start_method("forkserver")
        queue = mp.Queue()
        p = mp.Process(target = self.wrapper_func, args = [func, queue] + list(args), kwargs = kwargs)
        start_time = time.time()
        print(f'>> start - {p.name} at {time.ctime(start_time)}')
        p.start()
        result = queue.get()
        p.join()
        end_time = time.time()
        print(f'>> join - {p.name} at {time.ctime(end_time)}')
        if self.print_spend_time:
            t = end_time - start_time
            print(f">> {p.name} spend {round(t//3600)}h {round((t%3600)//60)}m {round(t%60,1)}s")
        print()
        return result