from module import ConvertWav
import os
import librosa

dir_path = "./practice_data/"
word_folder = os.listdir(dir_path)

data = []
data_path = []
for folder in word_folder:
    path = dir_path+folder+'/'
    if not os.path.isdir(path):
        continue
    file_list = [file for file in os.listdir(path) if file.endswith(".wav")]
    for each in file_list[0:min(100, len(file_list))]:
        data_path.append(path+each)
        file = open(data_path[-1], 'rb')
        l1 = file.readline()
        sr = int(ConvertWav(l1[24: 28], 'little'), 16)
        data.append(librosa.load(path+each, sr=sr))
        file.close()

