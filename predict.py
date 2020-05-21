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
import json

import utils

ap = argparse.ArgumentParser(description='Predict.py')

ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu_enabled', dest="gpu_enabled",type = bool, action="store", default="True")
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--checkpoint', dest="checkpoint", action="store", default="checkpoint.pth")
ap.add_argument('--img_path', dest="img_path", action="store", default="flowers/test/100/image_07896.jpg")




args = ap.parse_args()
image_path = args.img_path
architecture = args.arch
top_k = args.top_k
gpu_enabled = args.gpu_enabled
checkpoint_path = args.checkpoint
category_names = args.category_names


model  = utils.load_checkpoint(checkpoint_path,gpu_enabled)




with open(category_names, 'r') as f:
    cat_to_name = json.load(f)



probs, classes = utils.predict(image_path,model,top_k,cat_to_name)




for i in range(len(classes)):
    print('Flower name:{}, Probability:{}'.format(classes[i],probs[0][i].tolist()))
          