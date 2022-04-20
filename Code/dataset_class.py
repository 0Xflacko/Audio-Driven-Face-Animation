from __future__ import annotations, print_function, division
from email.mime import audio
import os
from re import I
from signal import signal
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image
from csv import reader



# Ignore warnings,
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
s_path = r"C:\Users\dinok\Documents\Dissertetion\Dataset"

class ImageAudioDatasetClass(Dataset):

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.traget_transform = target_transform
        self.images= []
        self.audio = []
        with open(annotations_file,'r') as csvfile:
            csv_reader = reader(csvfile)
            i=0

            for row in csv_reader:    
                for i in range(int(row[2])):
                    path_image = f"{row[0]}"%i
                    path_audio = f"{row[1]}"%i
                    
                    if os.path.exists(path_audio) and os.path.exists(path_image):       
                        image = read_image(path_image)
                        self.images.append(image)
                        signal = np.load(path_audio)
                        self.audio.append(signal) 
                        
                        
                        #print(path_image, path_audio)
                                
                    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):   
        return self.images[idx], self.audio[idx]        
        

face_dataset = ImageAudioDatasetClass(annotations_file = 'Dataset/new.csv', img_dir = s_path )
audio= face_dataset[0]