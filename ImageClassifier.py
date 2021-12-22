from PIL import Image
from torch.utils.data import Dataset
import os
from glob import glob
import torch
import torchvision.transforms as transforms
import numpy as np
import math


# Dataset Class for Setting up the data loading process
class inaturalist(Dataset):
    def __init__(self, root_dir, mode = 'train', transform = True):
        self.data_dir = root_dir
        self.mode = mode
        self.transforms = transform
        self._init_dataset()
        if transform:
            self._init_transform()

    def _init_dataset(self):
        self.files = []
        self.labels = []
        dirs = sorted(os.listdir(os.path.join(self.data_dir, 'train')))
        dirs = [dir for dir in dirs if dir[0]!= "."]
        if self.mode == 'train':
            for dir in range(len(dirs)):
                files = sorted(glob(os.path.join(self.data_dir, 'train', dirs[dir], '*.jpg')))
                self.labels += [dir]*len(files)
                self.files += files
        elif self.mode == 'val':
            for dir in range(len(dirs)):
                files = sorted(glob(os.path.join(self.data_dir, 'val', dirs[dir], '*.jpg')))
                self.labels += [dir]*len(files)
                self.files += files
        else:
            print("No Such Dataset Mode")
            return None

    def _init_transform(self):
        self.transform = transforms.Compose([
            transforms.Resize((256,256)),transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  #converts image to PyTorch tensor and resizes it
        ])

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert('RGB')
        label = self.labels[index]

        if self.transforms:
            img = self.transform(img)

        label = torch.tensor(label, dtype = torch.long)

        return img, label

    def __len__(self):
        return len(self.files)





import torch
import torch.nn as nn
import torch.nn.functional as F

#Class to define the model which we will use for training
def conv_block(in_channels,out_channels,pool=False):
    layers = [nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1), nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2,2))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, n_classes, in_channels=3):
        super().__init__()


        self.conv1 = conv_block(in_channels,32)
        self.conv2 = conv_block(32,64)
        self.conv3 = conv_block(64,64)

        self.pool = nn.MaxPool2d(2,2)

        self.conv4 = conv_block(64,128)
        self.conv5 = conv_block(128,128)
        self.conv6 = conv_block(128,128)

        self.conv7 = conv_block(128,256)
        self.conv8 = conv_block(256,256)
        self.conv9 = conv_block(256,256)

        self.conv10 = conv_block(256,512)
        self.conv11 = conv_block(512,512)
        self.conv12 = conv_block(512,512)

        self.conv13 = conv_block(512,1024)
        self.conv14 = conv_block(1024,1024)
        self.conv15 = conv_block(1024,1024)

        self.conv16 = conv_block(1024,2048)
        self.conv17 = conv_block(2048,2048)
        self.conv18 = conv_block(2048,2048)

        self.adapool = nn.AdaptiveAvgPool2d(output_size = (1,1))

        self.bridge = nn.Sequential(nn.Flatten(), nn.Linear(2048,n_classes))
        self.norm1 = nn.BatchNorm1d(512)
        self.finalline = nn.Linear(512,n_classes)





    def forward(self,xb):
        res = self.conv1(xb)
        x = self.conv2(res)
        x = self.conv3(x)
        x = self.pool(x)

        res = self.conv4(x)
        x = self.conv5(res)
        x = (self.conv6(x) + res)/math.sqrt(2)
        x = self.pool(x)

        res = self.conv7(x)
        x = self.conv8(res)
        x = (self.conv9(x) + res)/math.sqrt(2)
        x = self.pool(x)

        res = self.conv10(x)
        x = self.conv11(res)
        x = (self.conv12(x) + res)/math.sqrt(2)
        x = self.pool(x)

        res = self.conv13(x)
        x = self.conv14(res)
        x = (self.conv15(x) + res)/math.sqrt(2)
        x = self.pool(x)

        res = self.conv16(x)
        x = self.conv17(res)
        x = (self.conv18(x) + res)/math.sqrt(2)
        x = self.pool(x)

        x = self.adapool(x)

        x = self.bridge(x)
        #x = nn.ReLU(x)
        #x = self.norm1(x)
        #x = self.finalline(x)
        return F.softmax(x,dim=1)






#from dataloader import inaturalist
#from model import Classifier
import torch.nn as nn
import torch.optim as optim
import os
import time
import torch
from torch.utils.data import DataLoader
#from torchinfo import summary

############################################# DEFINE HYPERPARAMS #####################################################
# Feel free to change these hyperparams based on your machine's capactiy
batch_size = 8
epochs = 40
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################# DEFINE DATALOADER #####################################################
trainset = inaturalist(root_dir='../input/inaturalist12k/Data/inaturalist_12K', mode='train')
valset = inaturalist(root_dir='../input/inaturalist12k/Data/inaturalist_12K', mode = 'val')

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
valloader = DataLoader(valset, batch_size=2, shuffle=False, num_workers=0)

################################### DEFINE LOSS FUNCTION, MODEL AND OPTIMIZER ######################################
model=ResNet(10).to(device)   ##creating model object
criterion = nn.CrossEntropyLoss()    #loss function here is cross entropy (classification problem)
optimizer=torch.optim.Adam(model.parameters(),learning_rate)   #optimizer here is Adam
################################### CREATE CHECKPOINT DIRECTORY ####################################################

# NOTE: If you are using Kaggle to train this, remove this section. Kaggle doesn't allow creating new directories.
#checkpoint_dir = 'Documents/checkpoints'
#if not os.path.isdir(checkpoint_dir):
    #os.makedirs(checkpoint_dir)

#################################### HELPER FUNCTIONS ##############################################################

#def get_model_summary(model, input_tensor_shape):
   #summary(model, input_tensor_shape)

def accuracy(y_pred, y):
    _, predicted = torch.max(y_pred.data, 1)
    total = y.size(0)
    correct = (predicted == y).sum().item()
    return correct/total

def train(model, dataset, optimizer, criterion, device):

    model.train()
    netloss=0
    netacc=0
    for i, (images,labels) in enumerate(dataset):
#         if i%40==0:
#             print("round ",i)
        optimizer.zero_grad()      ##sets gradients to zero again to prevent any incorrect calculation
        images, labels = next(iter(dataset))      ##loading the data accordingly
        images, labels = images.to(device), labels.to(device)
        preds = model.forward(images)     ##forward pass
        loss=criterion(preds,labels)      ##calculate loss
        loss.backward()              ##gradient descent
        optimizer.step()             ##optimizer updates the parameters
        acc=accuracy(preds,labels)   ##calculate accuracy
        #print("Loss: {:.4f} \nAccuracy: {:.4f}".format(loss,acc))
        netloss=netloss+loss.item()
        netacc=netacc+acc
        #print(netloss,netacc)
        if((i+1)%10==0):
            print("Step: {}/{}, Loss: {:.4f}, Accuracy: {:.4f}%".format(i+1, len(dataset), netloss/(i+1), 100*netacc/(i+1)), end = '\r')

    #print("Loss: {:.4f} \nAccuracy: {:.4f}".format(netloss/len(dataset),netacc/len(dataset)))
    train_loss.append(netloss)
    train_acc.append(netacc)


def eval(model, dataset, criterion, device):

    model.eval()
    netloss=0
    netacc=0
    for i, (images,labels) in enumerate(dataset):
        #images, labels = next(iter(dataset))      ##loading the data accordingly
        images, labels = images.to(device), labels.to(device)
        preds= model.forward(images) ##forward pass
        loss=criterion(preds, labels)  ##calculate loss
        acc=accuracy(preds, labels)  ##calculate accuracy
        #print("Loss: {:.4f} \nAccuracy: {:.4f}".format(loss,acc))
        netloss=netloss+loss.item()
        netacc=netacc+acc
        #print(netloss,netacc)
        if((i+1)%10==0):
            print("Step: {}/{}, Loss: {:.4f}, Accuracy: {:.4f}%".format(i+1, len(dataset), netloss/(i+1), 100*netacc/(i+1)), end = '\n')

    #print("Loss: {:.4f} \nAccuracy: {:.4f}".format(netloss/len(dataset),netacc/len(dataset)))
    val_loss.append(netloss)
    val_acc.append(netacc)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

################################################### TRAINING #######################################################

# Get model Summary
#get_model_summary(model, [batch_size, 3, 256, 256])

#Training and Validation
best_valid_loss = float('inf')
train_loss = []
train_acc = []
val_loss = []
val_acc = []

#evalfn(model,valloader,criterion,device)

for epoch in range(epochs):

    start_time = time.monotonic()

    train(model,trainloader,optimizer,criterion,device)
    eval(model,valloader,criterion,device)
    torch.save(model.state_dict(), 'weights_only.pth')
    torch.save(model, 'entire_model.pth')

    end_time = time.monotonic()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print("\nTIME TAKEN FOR EPOCH {}: {} mins and {} seconds".format(epoch, epoch_mins, epoch_secs))


print("OVERALL TRAINING COMPLETE")
