import numpy as np
import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt

rate = 44100

input_txt_file_name = 'wave_log.txt'
f = open(input_txt_file_name, 'r')
line = f.readline()
l=[]
while line:
    l.append(line) 
    line = f.readline()
wav_data = np.asarray(l, dtype=np.float32)
f.close

wav_data = (wav_data/np.amax(wav_data))*32767
wav_data = np.asarray(wav_data, dtype=np.int16)
output_wav_file_name = "convert.wav"
scipy.io.wavfile.write(output_wav_file_name,rate,wav_data)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(wav_data)
output_png_file_name = "convert.png" 
fig.savefig(output_png_file_name) 
