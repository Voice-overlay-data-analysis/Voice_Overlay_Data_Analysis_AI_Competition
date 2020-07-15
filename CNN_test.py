from module import ConvertWav
import os
import librosa
import matplotlib.pyplot as plt
import time
from random import shuffle, seed
import numpy as np
from sklearn.model_selection import train_test_split

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
    for each in file_list[0:min(100, len(file_list))]:
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
#Split Dataset
#======================================================

data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.3, stratify=label, random_state=34)

#======================================================
#Data Preprocessing(MFCC)
#======================================================

mfcc_train = []
mfcc_test = []
for i in range(len(data_train)):
    mfcc_train.append(librosa.feature.mfcc(y=data_train[i], sr=sr))
for i in range(len(data_test)):
    mfcc_test.append(librosa.feature.mfcc(y=data_test[i], sr=sr))

pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a.np.zeros((a.shape[0], i-a.shape[i]))))

#======================================================
#Save mfcc/label data before Modeling
#======================================================

np.save("./practice_data_preprocessing_complete/mfcc_train.npy", mfcc_train)
np.save("./practice_data_preprocessing_complete/mfcc_test.npy", mfcc_test)
np.save("./practice_data_preprocessing_complete/label_train.npy", label_train)
np.save("./practice_data_preprocessing_complete/label_test.npy", label_test)


