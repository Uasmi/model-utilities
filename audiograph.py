import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt
import wave
import matplotlib
from scipy.fftpack import fft 
import audioop
import time
Chunk = 1024


p = pyaudio.PyAudio()
p.get_default_input_device_info()
wf = wave.open('inst16b.wav','rb')

stream = p.open(
    format = p.get_format_from_width(wf.getsampwidth()),
    channels = wf.getnchannels(),
    rate = wf.getframerate(),
    output = True
    )

start = time.time()
audioData = wf.readframes(Chunk)
while audioData != '':
    stream.write(audioData)
    audioData = wf.readframes(Chunk)
    #print (audioData)
    amplitude = audioop.rms(audioData, 2)
    #print (amplitude)
    dataInt = struct.unpack(str(4 * Chunk) + 'B', audioData)
    #print (dataInt)
    fftData = fft(dataInt)
    fftArray = np.abs(fftData[0: Chunk]) * 2 / (256 * Chunk)
    fftArray = fftArray[1: 10]
    fftArray *= 10

    #blockLinearRms= np.sqrt(np.mean(dataInt**2))

    #print (fftArray)
    if (amplitude > 15000) & (time.time() - start > 0.3):
        start = time.time()
        print("bang")