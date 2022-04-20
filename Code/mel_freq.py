#THIS PART IS NOT COMPLETED YET



import matplotlib.pyplot as plt
import numpy as np
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from python_speech_features import fbank
from python_speech_features import lifter
from scipy.fftpack import dct
import python_speech_features
import scipy.io.wavfile as wav
import torch
import os
from skimage import io
import collections
import matplotlib.pyplot as plt
import librosa
import librosa.display

 

        
dirloc = r"C:\Users\dinok\Documents\Dissertetion\Dataset\Seperated"
s_path = r"C:\Users\dinok\Documents\Dissertetion\Dataset\clip\images"

#Sound Processing
for file in os.scandir(dirloc):    
    if(file.path.endswith(".wav")) and file.is_file():

        j = 0
        (rate,sig) = wav.read(file.path)
        print(file)

        winstep=0.0167

        #calculate_nfft(41100, winlen=0.025)
        mfcc_extract = mfcc(sig, rate, winlen=0.025, winstep=0.0167, numcep=13, nfilt=26, nfft = 1536,
                            lowfreq=0, highfreq= None, preemph=0.97, ceplifter=22, appendEnergy=True, winfunc=lambda x:np.ones((x,)))


        #Visualize Mfcc with Librosa Lib
        plt.figure(figsize=(25,10))
        librosa.display.specshow(mfcc_extract,
                                x_axis='time',
                                sr=rate)

        plt.colorbar(format="%+2f")
        plt.show()


        #log_fbank_feat = logfbank(sig,rate, winlen = 0.025, winstep = 0.0167, nfilt = 26, nfft = 1536, lowfreq = 0, highfreq = None, preemph = 0.97)
        print(mfcc_extract.shape)





        k = 0
    
       # for k in range(len(mfcc_extract)):

            #//SAVING PART//UNCOMMENT TO SAVE NPY FILES
        #    name = './Dataset/' + file.name + '/audio/%05d.npy'%j
         #   print('Creating...' + name)
          #  os.makedirs('./Dataset/' + file.name + '/audio/', exist_ok=True)
           # np.save(name, mfcc_extract[k,:], allow_pickle=True, fix_imports=True)
           # j += 1
           # k += 1




            #Saving log_fbank_feat
            #name = './Dataset/' + file.name + '/audio/%05d.npy'%j
            #print('Creating...' + name)
            #np.save(name, log_fbank_feat, allow_pickle = True, fix_imports = True)
            #j +=1  
            #print(fbank_feat[1:3,:])