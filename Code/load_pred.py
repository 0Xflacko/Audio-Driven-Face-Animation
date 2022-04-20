from ctypes import sizeof
from re import X
import os
from unittest import result
from PIL import Image
import matplotlib
from torch.nn.functional import normalize
from matplotlib import pyplot as plt
from torch import nn, tensor
import torch
import numpy as np
from collections import OrderedDict
import cv2
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset_class import ImageAudioDatasetClass
import tensorflow as tf
from skimage import exposure
import scipy.misc




if __name__ == '__main__':
  device = "cuda" if torch.cuda.is_available() else "cpu"
  s_path = r"C:\Users\dinok\Documents\Dissertetion\Dataset"

  print(f"Using {device} device")

  #Reshape Class
  class View(nn.Module):
    def __init__(self, size):
      super().__init__()
      self.size = size

    def forward(self,x):
      x = x.view(self.size)
      return x

  
  #Definition of Neural-Net
  class NeuralNet(nn.Module):
      def __init__(self):
          super(NeuralNet, self).__init__()
          self.linear_relu_stack = nn.Sequential(OrderedDict([
            ('lin1',  nn.Linear(13,32)),
            ('tanh1', nn.Tanh()),
            ('lin2',  nn.Linear(32,256)),
            ('relu2', nn.ReLU()),
            ('lin3',  nn.Linear(256,1024)),
            ('relu3', nn.ReLU()),
            ('lin4',  nn.Linear(1024,2048)),
            ('vew1',  View(size=(-1, 128, 4, 4))),
            ('tconv1',nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=2, stride= 2)),#For RBG in channels will usually be 3
            ('relu4', nn.ReLU()),
            ('tconv2',nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2)),
            ('relu5', nn.ReLU()),
            ('tconv3',nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=2,stride=2)),
            ('relu6', nn.ReLU()),
            ('tconv4',nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=2,stride=2)),
            ('relu7', nn.ReLU()),
            ('tconv5',nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=2,stride=2)),
            ('relu8', nn.ReLU()),
            ('tconv6',nn.ConvTranspose2d(in_channels=16,out_channels=3,kernel_size=2,stride=2)),
            ('relu9', nn.ReLU()),
          ]))
          
      def forward(self, x):
          logits = self.linear_relu_stack(x)
          return logits

  (face_dataset) = ImageAudioDatasetClass(annotations_file = 'Dataset/test.csv', img_dir = s_path )

  

  #Putting face_dataset into the DataLoader
  test = DataLoader(
    dataset = face_dataset,shuffle=True,num_workers=4,pin_memory=True
  )



  #Prediction & Saving loop
  with torch.no_grad():
      #Loading the model
    FILE = "model.pth"
    model = NeuralNet()
    print("=>LOADING:", FILE)
    state_dict = torch.load(FILE)
    model.load_state_dict(state_dict)
    model.eval()
      
    
    batch = 1
      
  # for batch, (X,y) in enumerate(test_dataloader):
    #    pred = model(y.float())
    #   print(type(pred))
      #  print(pred.size())
      # for j in range(len(pred)):
      #   name = './prediction/%04d.png'%j
      #   print('Creating...' + name)
      #   os.makedirs('./prediction/', exist_ok=True)
      #   pred = pred.detach()
      #   plt.imsave(name,pred[j].permute(1, 2, 0).numpy()/255.0)
      #   j += 1
          #plt.imsave(name,pred[j])
          #pred = torch.argmax(pred)
          #pred = pred.detach().numpy()
        # cv2.imwrite(name, pred)
    def NormalizeData(data):
      return (data - np.min(data)) / (np.max(data) - np.min(data))

    j = 0
    loc = r"C:\Users\dinok\Documents\Dissertetion\Dataset\video10\audio"

    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    
    for X, y in test: 
        pred = model(y.float())
        #print(pred.size())
        #pred = pred.detach().numpy()
        pred = pred.detach()
        
        #for file in os.scandir(loc):
         #     if(file.path.endswith(".npy")) and file.is_file():
            #    a = []
            #   a = np.load(file)
              #  a = torch.tensor(a)
        name = './prediction/%04d.png'%j
        print('Creating...' + name)
        os.makedirs('./prediction/', exist_ok=True)  
                #pred[j] = normalize(pred[j], p=0.0, dim=0)
                #pred[j] = matplotlib.colors.Normalize(vmin = np.min(pred),vmax= np.max(pred))
                #pred = torch.tensor(pred)
                #print(pred.shape)
        
                #result = np.transpose(pred.argmax(1), (1,2,0))
                #result = np.squeeze(result, axis = 0) 
                #print(result.shape)
                #print(pred.argmax(1).shape)
                #NormalizeData(result)
              # result = exposure.rescale_intensity(result)
                #plt.imsave(name,result/255.0)
       # pred = torch.squeeze(pred, 0)
       # print(pred[j].shape)
       # plt.imsave(name, pred.permute(1, 2, 0).numpy()/255.0)
        
        pred = pred.reshape((3, pred.shape[2], pred.shape[3]))
        pred = pred.permute(1, 2, 0)
        print(pred.shape)
        
        #pred = cv2.filter2D(src=pred.numpy(), ddepth=-1, kernel=kernel)
        pred = pred.numpy()
        cv2.imwrite(name, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))  
        j += 1
    


    
      
  
    