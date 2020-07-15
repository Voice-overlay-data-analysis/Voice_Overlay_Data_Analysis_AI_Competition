from module import ConvertWav
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
#import tensorflow as tf
#from practice_model_fn import model_fn, train_fn, eval_fn, test_fn

#======================================================
#Load Dataset
#======================================================

dir_path = "./practice_data/"
word_folder = os.listdir(dir_path)

data = []
data_path = []
label = []
i = -1
for folder in word_folder:
    path = dir_path+folder+'/'
    if not os.path.isdir(path):
        continue
    i += 1
    file_list = [file for file in os.listdir(path) if file.endswith(".wav")]
    for each in file_list[0 : min(100, len(file_list))]:
        data_path.append(path+each)
        file = open(data_path[-1], 'rb')
        l1 = file.readline()
        res, sr = librosa.load(path+each, sr=int(ConvertWav(l1[24: 28], 'little'), 16))
        data.append(res)
        label.append(i)
        file.close()

data = np.array(data)
label = np.array(label)
data_path = np.array(data_path)

#======================================================
#Data Preprocessing(MFCC) & Data Padding
#======================================================

mfcc = []

max_padding = 0
for i in range(len(data)):
    res = librosa.feature.mfcc(y=data[i], sr=sr)
    if max_padding < res.shape[1]:
        max_padding = res.shape[1]
    mfcc.append(res)

#Data padding
for i in range(len(mfcc)):
    mfcc[i] = np.hstack((mfcc[i], np.zeros((mfcc[i].shape[0], max_padding-mfcc[i].shape[1]))))
#end

mfcc = np.array(mfcc)

#======================================================
#Save data before Modeling
#======================================================

np.save("./practice_data_preprocessing_complete/data.npy", data)
np.save("./practice_data_preprocessing_complete/mfcc.npy", mfcc)
np.save("./practice_data_preprocessing_complete/label.npy", label)

#======================================================
#Load data for Modeling
#======================================================

data_path = "./practice_data_preprocessing_complete/"

mfcc = np.load(data_path+'mfcc.npy')
label = np.load(data_path+'label.npy')

#======================================================
#Split Dataset
#======================================================

mfcc, mfcc_test, label, label_test = train_test_split(mfcc, label, test_size=0.2, stratify=label, random_state=34)
mfcc_train, mfcc_eval, label_train, label_eval = train_test_split(mfcc, label, test_size=0.25, stratify=label, random_state=34)

# > train : eval : test = 3: 1: 1

#======================================================
#Modeling
#======================================================

dir_path = "./practice_model_res/"

est = tf.estimator.Estimator(model_fn, model_dir=dir_path)
est.train(train_fn)
valid = est.evaluate(eval_fn)
predict = est.predict(test_fn)

#======================================================