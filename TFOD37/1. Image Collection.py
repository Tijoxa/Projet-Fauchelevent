## 1. Import Dependencies
!pip install opencv-python

import cv2
import uuid
import os
import time

## 2. Define Images to Collect
labels = ['Stade1', 'Stade2', 'Stade3', 'Stade4']
number_imgs = 5

## 3. Setup Folders
IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')

if not os.path.exists(IMAGES_PATH):
    if os.name == 'posix':
        !mkdir -p {IMAGES_PATH}
    if os.name == 'nt':
         !mkdir {IMAGES_PATH}
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        !mkdir {path}

## 4. Image Labelling
!pip install --upgrade pyqt5 lxml

LABELIMG_PATH = os.path.join('Tensorflow', 'labelimg')

if not os.path.exists(LABELIMG_PATH):
    !mkdir {LABELIMG_PATH}
    !git clone https://github.com/tzutalin/labelImg {LABELIMG_PATH}

if os.name == 'posix':
    !make qt5py3
if os.name =='nt':
    !cd {LABELIMG_PATH} && pyrcc5 -o libs/resources.py resources.qrc

!cd {LABELIMG_PATH} && python labelImg.py

## 6. Move them into a Training and Testing Partition

## OPTIONAL - 7. Compress them for Colab Training
TRAIN_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'train')
TEST_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'test')
ARCHIVE_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'archive.tar.gz')

!tar -czf {ARCHIVE_PATH} {TRAIN_PATH} {TEST_PATH}