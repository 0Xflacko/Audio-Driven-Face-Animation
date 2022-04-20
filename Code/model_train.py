import os
from re import X
import sys
from collections import OrderedDict
from email.mime import audio
from multiprocessing import freeze_support
from matplotlib import image
import numpy as np
from signal import signal
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
import torchvision.models as models
from typing_extensions import Self
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
from dataset_class import ImageAudioDatasetClass
from torchsummary import summary
import torch.optim.lr_scheduler



if __name__ == '__main__':
  device = "cuda" if torch.cuda.is_available() else "cpu"
  s_path = r"C:\Users\dinok\Documents\Dissertetion\Dataset"

  print(f"Using {device} device")
  (face_dataset) = ImageAudioDatasetClass(annotations_file = 'Dataset/new.csv', img_dir = s_path )

  #Size Class
  class PrintSize(nn.Module):
    def __init__(self):
      super(PrintSize, self).__init__()

    def forward(self,x):
      print(x.shape)
      return x

  #Reshape Class
  class View(nn.Module):
    def __init__(self, size):
      super().__init__()
      self.size = size

    def forward(self,x):
      x = x.view(self.size)
      return x

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
            #('relu9', nn.ReLU()),
          ]))
          
      def forward(self, x):
          logits = self.linear_relu_stack(x)
          return logits
          

  #Splitting dataset into train,test:
  train, test = train_test_split(face_dataset,test_size=0.5)

 
  #Putting face_dataset into the DataLoader
  train_dataloader = DataLoader(
    dataset = train,shuffle=True,num_workers=4,pin_memory=True
  )

  test_dataloader = DataLoader(
    dataset = test, shuffle=True,num_workers=4,pin_memory=True
  )

  #Learning rate
  learning_rate = 0.0005
  batch_size = 1
  batch = 1
  epochs = 5

  #print(train[0])
  #print(test[0])

  model = NeuralNet()

  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9) #Tuned FOR MAX Performance

  # Defining the Training Loop
  #@torch.jit.script
  def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    for batch, (X,y) in enumerate(dataloader):
      # Compute prediction and loss
      optimizer.zero_grad(set_to_none= True) #Sets all gradients to zero for frame, TUNED FOR BETTER PERFORMANCE
      pred = model(y.float())
      loss = loss_fn(pred,X.float())

      # Backpropagation
      
      loss.backward()
      
      #Gradient Descent or adam step 
      optimizer.step()
      
      
      if batch % 100 == 0:
        loss, current = loss.item(), batch * len(y)
        print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
      #Tuned for Max Performance
    scheduler.step()


  # Testing Loop
  
  def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
      for X, y in dataloader:
        pred = model(y.float())
        test_loss += loss_fn(pred,X.float()).item()
        correct += (pred.argmax(1) == X).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

  epochs = 10
  for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    torch.backends.cudnn.benchmark = True #TUNED FOR MAX PERFORMANCE
    train_loop(train_dataloader,model,loss_fn,optimizer)
    test_loop(test_dataloader,model,loss_fn)
    
  print("Done!")

  #Saving the best model
  FILE = "model.pth"
  torch.save(model.state_dict(),FILE)
  
