# Imports here
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
from collections import OrderedDict
import argparse

import utils


ap = argparse.ArgumentParser(description='Train.py')




ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--learning_rate', dest="learning_rate", action="store",type=float, default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store",type=float, default = 0.3)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=15)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
ap.add_argument('--gpu_enabled', dest="gpu_enabled",type = bool, action="store", default="True")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")


args = ap.parse_args()
architecture = args.arch
learning_rate = args.learning_rate
dropout = args.dropout
epochs = args.epochs
hidden_units = args.hidden_units
gpu_enabled = args.gpu_enabled
path = args.save_dir


print(architecture,learning_rate,dropout,epochs,hidden_units,gpu_enabled,path)

train_dataloader, valid_dataloader, test_dataloader, train_datasets= utils.set_data()  


model, criterion, optimizer, device = utils.set_up(architecture,hidden_units,learning_rate,dropout,gpu_enabled)


model, optimizer  = utils.train_model(epochs,dropout, model, criterion, optimizer, device, train_dataloader, valid_dataloader)


def accuracy():
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader: 
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print("Test accuracy is: "+"{:.2%}".format(correct / total))
    
accuracy()


utils.save_checkpoint(path,hidden_units,dropout,architecture,learning_rate,optimizer,model,train_datasets)


print('Model checkpoint saved')
print('Done!!!......')



