# Image Classifier 

  Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smartphone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.
  
  In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice, you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories.
  
  When you've completed this project, you'll have an application that can be trained on any set of labelled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset.
  
  *This is the final Project of the Udacity AI with Python Nanodegree*


## Prerequisites
The Code is written in Python 3.6.5 . If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. 

To install pip run in the command Line
```
python -m ensurepip -- default-pip
``` 
to upgrade it 
```
python -m pip install -- upgrade pip setuptools wheel
```
to upgrade Python
```
pip install python -- upgrade
```
Additional Packages that are required are: [Numpy](http://www.numpy.org/), [Pandas](https://pandas.pydata.org/), [MatplotLib](https://matplotlib.org/), [Pytorch](https://pytorch.org/), PIL and json.\
You can donwload them using [pip](https://pypi.org/project/pip/)
```
pip install numpy pandas matplotlib pil
```
or [conda](https://anaconda.org/anaconda/python)
```
conda install numpy pandas matplotlib pil
```
In order to intall Pytorch head over to the Pytorch site select your specs and follow the instructions given.

## Viewing the Jyputer Notebook
In order to better view and work on the jupyter Notebook I encourage you to use [nbviewer](https://nbviewer.jupyter.org/) . You can simply copy and paste the link to this website and you will be able to edit it without any problem. Alternatively you can clone the repository using 
```
git clone https://github.com/AyusmaTech/Image-Classifier.git
```
then in the command Line type, after you have downloaded jupyter notebook type
```
jupyter notebook
```
locate the notebook and run it.


## Command Line Application
* Train a new network on a data set with ```train.py```
  * Basic Usage : ```python train.py ```
  * Prints out current epoch, training loss, validation loss, and validation accuracy as the netowrk trains
  * Options:
    * Set direcotry to save checkpoints: ```python train.py --save_dir checkpoint.pth```
    * Choose arcitecture: ```pytnon train.py --arch vgg16```
    * Set hyperparameters: ```python train.py  --learning_rate 0.001 --hidden_units 120 --epochs 15 --dropout 0.3  ```
    * Use GPU for training: ```python train.py  --gpu_enabled True```
    
* Predict flower name from an image with ```predict.py``` along with the probability of that name. That is you'll pass in a single image ```/path/to/image``` and return the flower name and class probability
  * Basic usage: ```python predict.py ```
  * Options:
    * Return top **K** most likely classes:``` python predict.py --top_k 5```
    * Choose image for Prediction: ```python predict.py --img_path flowers/test/100/image_07896.jpg```
    * Choose checkpoint to use for prediction:``` python predict.py --checkpoint checkpoint.pth```
    * Use a mapping of categories to real names: ```python predict.py --category_names cat_to_name.json```
    * Use GPU for inference: ```python predict.py --gpu_enabled True```
    
