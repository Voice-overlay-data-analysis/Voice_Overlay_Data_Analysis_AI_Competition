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

l2 = file.readline()