#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 09:13:13 2020

@author: operator
"""

# Install packages
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
import os
import sys

# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
import cv2
import os
import numpy as np
from tqdm import tqdm
from keras.utils import np_utils

# Function to load images
def get_img_data(data_dir, labels):
    
    data = [] 
    
    # Iterate
    for label in labels: 
        
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        
        print('Loading images for {} class..'.format(label))
        
        for img in tqdm(os.listdir(path)):
            
            try:
                
                # Load
                img1 = cv2.imread(os.path.join(path, img))[...,::-1] 
                img2 = cv2.resize(img1, (224, 224)) 
                data.append([img2, class_num])
                
            except Exception as e:
                
                print(e)
                
    return np.array(data)

# Function to prep training set
def prep_img_set(img_set):
    
    # Initialize
    x, y = [], []
    
    # Iterate
    for feature, label in img_set:
  
        x.append(feature)
        y.append(label)

    # Normalize 
    x = np.array(x) / 255
    x.reshape(-1, 224, 224, 1)
    
    y = np.array(y)
    
    return x, y

# Function to build CNN
def build_img_net():

    # Initialize model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = xtrain.shape[1:], padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), input_shape = (3, 32, 32), activation = 'relu', padding = 'same'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())  
    model.add(Conv2D(128, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, kernel_constraint = maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(128, kernel_constraint = maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(class_num))
    model.add(Activation('softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model

# Execute
if __name__ == "__main__":
    
    # Navigate
    os.chdir(input('Enter path for wdir: '))
    #os.system('pip install requirements.txt')
    
    # Load
    print('Loading data..')
    
    labels = sys.argv[1].split()
    train = get_img_data('train', labels)
    val = get_img_data('val', ['PNEUMONIA', 'NORMAL'])

    xtrain, ytrain = prep_img_set(train)
    xval, yval = prep_img_set(val)
    print('Complete!\n')
    
    # One hot encode target
    ytrain = np_utils.to_categorical(ytrain)
    yval = np_utils.to_categorical(yval)
    class_num = yval.shape[1]
    
    # Initialize
    print('Building neural net..')
    nn = build_img_net()
    
    # Train
    nn.fit(xtrain, ytrain, validation_data = (xval, yval), epochs = 10, batch_size = len(ytrain))

    # Evaluate
    scores = model.evaluate(xtest, ytest, verbose = 0)
    print('\nImage Classification Neural Net complete!\n')
    print("Accuracy: %.2f%%" % (scores[1] * 100))
