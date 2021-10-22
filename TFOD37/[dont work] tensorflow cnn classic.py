##
import pandas as pd
import numpy as np

##
from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils.np_utils import to_categorical

##
import cv2
import os
import glob
import gc

def lire_images(img_dir, xdim, ydim, label, nmax=5000) :
    """
    Lit les images dans les sous répertoires de img_dir
    nmax images lues dans chaque répertoire au maximum
    Renvoie :
    X : liste des images lues, matrices xdim*ydim
    y : liste des labels numériques
    """
    X = []
    y = []

    for dirname in os.listdir(img_dir):
        data_path = os.path.join(img_dir + "/" + dirname)
        files = glob.glob(data_path)
        n = 0
        for f1 in files:
            if n>nmax : break
            img = cv2.imread(f1)
            # img = cv2.resize(img, (xdim,ydim))
            X.append(np.array(img))
            y.append(label)
            n += 1
    X = np.array(X)
    y = np.array(y)
    gc.collect()  # Récupération de mémoire
    return X,y

## Epi
Xtrain, ytrain = lire_images("C:/Users/Tijoxa/Desktop/code/dataset Global Wheat/train/Epi", 1024, 1024, 'Epi', 2000)

##
Xtest, ytest = lire_images("C:/Users/Tijoxa/Desktop/code/dataset Global Wheat/test/Epi", 1024, 1024, 'Epi', 2000)

## Tallage
Xtemp, ytemp = lire_images("C:/Users/Tijoxa/Desktop/code/dataset Global Wheat/train/Tallage", 1024, 1024, 'Tallage', 2000)
Xtrain = np.concatenate((Xtrain, Xtemp))
ytrain = np.concatenate((ytrain, ytemp))

##
Xtemp, ytemp = lire_images("C:/Users/Tijoxa/Desktop/code/dataset Global Wheat/test/Tallage", 1024, 1024, 'Tallage', 2000)
Xtest = np.concatenate((Xtest, Xtemp))
ytest = np.concatenate((ytest, ytemp))

## Levée
Xtemp, ytemp = lire_images("C:/Users/Tijoxa/Desktop/code/dataset Global Wheat/train/Levee", 1024, 1024, 'Levee', 2000)
Xtrain = np.concatenate((Xtrain, Xtemp))
ytrain = np.concatenate((ytrain, ytemp))

##
Xtemp, ytemp = lire_images("C:/Users/Tijoxa/Desktop/code/dataset Global Wheat/test/Levee", 1024, 1024, 'Levee', 2000)

Xtest = np.concatenate((Xtest, Xtemp))
ytest = np.concatenate((ytest, ytemp))

## Moisson
Xtemp ,ytemp = lire_images("C:/Users/Tijoxa/Desktop/code/dataset Global Wheat/train/Moisson", 1024, 1024, 'Moisson', 2000)
Xtrain = np.concatenate((Xtrain, Xtemp))
ytrain = np.concatenate((ytrain, ytemp))

##
Xtest, ytest = lire_images("C:/Users/Tijoxa/Desktop/code/dataset Global Wheat/test/Moisson", 1024, 1024, 'Moisson', 2000)
Xtest = np.concatenate((Xtest, Xtemp))
ytest = np.concatenate((ytest, ytemp))

##
# Réseau convolutionnel simple
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(1024, 1024, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

##
train = model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=2, batch_size=1, verbose=1)

##
y_cnn = model.predict_classes(Xtest)