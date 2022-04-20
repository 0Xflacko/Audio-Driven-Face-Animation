#THIS PART IS NOT COMPLETED YET



import matplotlib.pyplot as plt
import numpy as np
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import python_speech_features
import scipy.io.wavfile as wav
import torch
import os
from skimage import io
import collections

        
dirloc = r"C:\Users\dinok\Documents\Dissertetion\Dataset\Seperated"
s_path = r"C:\Users\dinok\Documents\Dissertetion\Dataset\clip\images"

#Sound Processing
for file in os.scandir(dirloc):    
    if(file.path.endswith(".wav")) and file.is_file():
        
        (rate,sig) = wav.read(file.path)
        
        mfcc_extract = mfcc(sig, rate, winlen=0.025, winstep=0.001, numcep=13, nfilt=26, nfft=512,
                            lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc= np.hamming)

        d_mfcc_feat = delta(mfcc_extract, 2)
        fbank_feat = logfbank(sig,rate)
        #print(fbank_feat[1:3,:])