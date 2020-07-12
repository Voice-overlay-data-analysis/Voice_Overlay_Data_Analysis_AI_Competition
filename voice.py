import numpy as np
from scipy.io import wavfile
from module import ConvertWav

fs, data = wavfile.read('./train_00000.wav')
# fs = 초당 샘플링 속도
# data = numpy 배열

blocks = [['ChunkID', 'big', 4],
          ['ChunkSize', 'little', 4],
          ['Format', 'big', 4],
          ['Subchunk1ID', 'big', 4],
          ['Subchunk1Size', 'little', 4],
          ['AudioFormat', 'little', 2],
          ['NumChannels', 'little', 2],
          ['SampleRate', 'little', 4],
          ['ByteRate', 'little', 4],
          ['BlockAlign', 'little', 2],
          ['BitsPerSample', 'little', 2],
          ['Subchunk2ID', 'big', 4],
          ['Subchunk2Size', 'little', 4]]

file = open('./train_00000.wav', 'rb')
l1 = file.readline()

offset = 0
wav_info = []
for i in range(len(blocks)):
    if i in [0, 2, 3, 11]:
        wav_info.append([blocks[i][0], l1[offset : offset+blocks[0][2]]])
    elif i == 1:
        wav_info.append([blocks[i][0], int(ConvertWav(l1[offset : offset+blocks[i][2]], blocks[i][1]),16) + 8])
    else:
        wav_info.append([blocks[i][0], int(ConvertWav(l1[offset : offset+blocks[i][2]], blocks[i][1]),16)])
    offset += int(blocks[i][2])

wav_info = np.array(wav_info).reshape(-1, 2)

import librosa, librosa.display
import matplotlib.pyplot as plt

data, sr = librosa.load('./train_00000.wav', sr=16000)
#sr = sampling rate, librosa는 별도로 sr를 설정하지 않으면 default로 22050
#len(data) = Subchunk2Size / BitsPerSample = 32000 / 2byte(16bits)

#원본음성파일 주파수
time = np.linspace(0, len(data)/sr, len(data)) # time axis
fig, ax1 = plt.subplots() # plot
ax1.plot(time, data, color = 'b', label='speech waveform')
ax1.set_ylabel("Amplitude") # y 축
ax1.set_xlabel("Time [s]") # x 축
plt.show()

#MFCC : 입력된 신호에서 소리의 특징을 추출하는 기법
#입력된 소리 전체를 대상으로 하는 것이 아니라, 일정 시간(구간)으로 나누어서
#이 시간에 대한 스펙트럼을 분석하여 특징 추출

mfcc = librosa.feature.mfcc(y=data, sr=sr)
print(mfcc.shape) # ( 20, number_of_frames)
plt.pcolor(mfcc) #히트맵(HeatMap) 그리기


D = librosa.amplitude_to_db(librosa.stft(data[:]), ref=np.max)
plt.plot(D.flatten())
plt.show()

