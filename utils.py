# Imports here
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import keep_awake
import numpy as np
from PIL import Image
from collections import OrderedDict


def set_data():
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir,transform = train_transforms) 
    test_datasets = datasets.ImageFolder(test_dir,transform = test_transforms) 
    valid_datasets = datasets.ImageFolder(valid_dir,transform = test_transforms) 

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_datasets,batch_size=64,shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_datasets,batch_size=64,shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_datasets,batch_size=64,shuffle=True)
    
    return train_dataloader, valid_dataloader, test_dataloader,train_datasets


def set_up(architecture = 'vgg16', hidden_units = 120,  learning_rate = 0.001, dropout = 0.3, gpu_enabled = True ):
    
    if architecture == 'vgg16':
       model = models.vgg16(pretrained=True)
    elif architecture == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif architecture == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print('Architecture not available for this set up')
        
    device = torch.device("cuda" if torch.cuda.is_available() and gpu_enabled else "cpu")

    for param in model.parameters():
        param.requires_grad = False
    

    classifier = nn.Sequential(OrderedDict([
                              ('dropout',nn.Dropout(dropout)),
                              ('fc1',nn.Linear(25088,hidden_units)),
                              ('relu',nn.ReLU()),
                              ('fc2',nn.Linear(hidden_units,90)),
                              ('relu2',nn.ReLU()),
                              ('fc3',nn.Linear(90,80)),
                              ('relu3',nn.ReLU()),
                              ('fc4',nn.Linear(80,102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(),lr = learning_rate)

    model.to(device)
    
    return model, criterion, optimizer, device
        


def train_model(epochs, dropout, model, criterion, optimizer, device, train_dataloader,valid_dataloader):
    # TODO: Build and train your network
    steps = 0
    running_loss = 0
    print_every = 5
    

    for epoch in keep_awake(range(epochs)):
        for images, labels in train_dataloader:
            steps += 1
        
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(images)
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
      
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in valid_dataloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps,labels)
                    
                        test_loss += batch_loss.item() 
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                  
            
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(valid_dataloader):.3f}.. "
                      f"Vaildation accuracy: {accuracy/len(valid_dataloader):.3f}")
                running_loss = 0
                model.train()
                      
    print('Training done')
    return model, optimizer
          
def save_checkpoint(path, hidden_units, dropout,arch,lr, optimizer, model,train_datasets):
    checkpoint = {'input_size': 25088,
              'output_size': 102,
              'dropout': dropout,
              'lr': lr,
              'arch': arch,   
              'hidden_units': [hidden_units,90,80],
              'optimzer_state': optimizer.state_dict,
              'class_to_idx':  train_datasets.class_to_idx,
              'state_dict': model.classifier.state_dict()}

    torch.save(checkpoint, path)
          
def load_checkpoint(filepath,gpu_enabled):
    checkpoint = torch.load(filepath)
    model,_,optimizer,_ = set_up(checkpoint['arch'],checkpoint['hidden_units'][0],checkpoint['lr'],checkpoint['dropout'],gpu_enabled)   
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1',nn.Linear(checkpoint['input_size'],checkpoint['hidden_units'][0])),
                          ('relu',nn.ReLU()),
                          ('dropout',nn.Dropout(checkpoint['dropout'])),
                          ('fc2',nn.Linear(checkpoint['hidden_units'][0],checkpoint['hidden_units'][1])),
                          ('relu2',nn.ReLU()),
                          ('fc3',nn.Linear(checkpoint['hidden_units'][1],checkpoint['hidden_units'][2])),
                          ('relu3',nn.ReLU()),
                          ('fc4',nn.Linear(checkpoint['hidden_units'][2],checkpoint['output_size'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
                           
    model.classifier.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    device = torch.device("cuda" if torch.cuda.is_available() and gpu_enabled else "cpu")
    model.to(device)

    
    
    optimizer = checkpoint['optimzer_state']
    
    return model
          
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    im.resize((256,256))
    processed_image = transforms.Compose([transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    return processed_image(im)
          
          
          
def predict(image_path, model, topk, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image = image.cuda().float()
    
 
    
    model.eval()
 
    
    with torch.no_grad():
        output = model(image)
        prob, idxs = torch.topk(output, topk)
       
    
        idxs = np.array(idxs)            
        idx_to_class = {val:key for key, val in model.class_to_idx.items()}
        classes = [idx_to_class[idx] for idx in idxs[0]]
        
        
        names = []
        for cls in classes:
            names.append(cat_to_name[str(cls)])
        
        
        
        return np.exp(prob), names
          
          
          
                    
                      
                      
                      
                      
    

    
    
    
    