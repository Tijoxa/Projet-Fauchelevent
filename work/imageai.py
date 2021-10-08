import numpy as np
import pandas as pd
import cv2
import os
import re
import tensorflow
from PIL import Image
from matplotlib import pyplot as plt

from imageai.CLassification import ImageClassification

model.setModelTypeAsResNet50()
model.setModelPath("C:/Users/TM/Documents/Files/ML/fauchlevent/resnet50_imagenet_tf.2.0.h5")
model.loadModel()

##create model with custom data
"""

format of the Y file : https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5

https://imageai.readthedocs.io/en/latest/custom/index.html
create a dataset folder, one train and one test folders in it

in the train folder, create a folder for each label, named by the name of the label, then put images in it

in the test folder, same

the trained json is somewhere

from imageai.Classification.Custom import ClassificationModelTrainer

model = ClassificationModelTrainer()
model.setModelTypeAsResNet50()
model.setModelPath("C:/Users/TM/Documents/Files/ML/fauchlevent/resnet50_imagenet_tf.2.0.h5")
model.setJsonPath("path of the json.json")
model.setDataDirectory("path of the data_directory", json_subdirectory = path of the folder where there is the json file)
model.trainModel(things in it)
model.loadModel()

"""

##prediction

predictions, probabilities = prediction.classifyImage('image1.jpg', result_count = 5, input_type = 'file')  # returns 2 python list