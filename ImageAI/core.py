import os

'''Current path for the API needs to be updated if the script is run on another device'''
path = "C:/Users/DL/Documents/Projet-Fauchelevent-Purjack-patch-1/ImageAI/"
os.chdir(path)

import sys
sys.path.append(path)

import json
import shutil
import time

"""
Those classification architecture are supported:
mobilenetv2 / densenet121 / resnet50 / inceptionv3 / efficientnetb7
"""
model_type = "resnet50"

def getModelType():
    return model_type

test_subdirectory = path + "dataset/validation"
train_subdirectory = path + "dataset/train"

def readJson():
    with open(os.path.join(path, "model_class.json"), "r") as json_file:
        dirdict = json.load(json_file)
        json_file.close()
    return dirdict

## Customize weights of classes for unbalanced dataset
def normalizeClassWeights(train_subdirectory):

    dirdict = readJson()
    length_label = [1.0 for i in range(len(dirdict))]

    for i in range(len(dirdict)):
        folder_name = train_subdirectory + "/" + dirdict['{}'.format(i)] + "/"
        length_label[i] = len([name for name in os.listdir(folder_name)])

    max_length_label = max(length_label)

    class_weight_custom = {i: 0 if (length_label[i]==0) else max_length_label/length_label[i] for i in range(len(dirdict))}  # the json file provides the mapping between the folder name and the number of its class

    return class_weight_custom

class_weight_custom = normalizeClassWeights(train_subdirectory + "/")

## Train
from imageaicustom import ClassificationModelTrainer

def trainModelFunction(model_type, dataset_directory = path + "dataset", json_subdirectory = path, train_subdirectory = None, test_subdirectory = None, num_experiments = 200, continue_from_model = None):

    dirdict = readJson()

    model_trainer = ClassificationModelTrainer()

    if (model_type == "mobilenetv2"):
        model_trainer.setModelTypeAsMobileNetV2()
    elif (model_type == "densenet121"):
        model_trainer.setModelTypeAsDenseNet121()
    elif (model_type == "resnet50"):
        model_trainer.setModelTypeAsResNet50()
    elif (model_type == "inceptionv3"):
        model_trainer.setModelTypeAsInceptionV3()
    elif (model_type == "efficientnetb7"):
        model_trainer.setModelTypeAsEfficientNetB7()

    model_trainer.setDataDirectory(dataset_directory, json_subdirectory=json_subdirectory, test_subdirectory = path + "dataset/validation", train_subdirectory = path + "dataset/train")

    '''Put batch_size = 1 for heavier models (resnet50, dense121) that raise Out of memory issues'''
    model_trainer.trainModel(num_objects=len(dirdict), num_experiments=num_experiments, enhance_data=True, batch_size=1, training_image_size=1024, class_weight_custom=class_weight_custom, continue_from_model=continue_from_model)


## Test
from imageaicustom import CustomImageClassification

def testModelFunction(model_type, folder_path, model_path):
    dirdict = readJson()
    prediction = CustomImageClassification()

    if (model_type == "mobilenetv2"):
        prediction.setModelTypeAsMobileNetV2()
    elif (model_type == "densenet121"):
        prediction.setModelTypeAsDenseNet121()
    elif (model_type == "resnet50"):
        prediction.setModelTypeAsResNet50()
    elif (model_type == "inceptionv3"):
        model_trainer.setModelTypeAsInceptionV3()
    elif (model_type == "efficientnetb7"):
        prediction.setModelTypeAsEfficientNetB7()

    prediction.setModelPath(model_path)
    prediction.setJsonPath(path + "model_class.json")
    prediction.loadModel(num_objects = len(dirdict))

    preds={}
    for file in os.listdir(folder_path):
        file_path = folder_path+file
        preds[file]=prediction.classifyImage(file_path, result_count=1)
    return preds

## Confusion matrix
from tensorflow.math import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def confusionMatrix(model_file):
    '''returns nothing, but shows the diagram'''
    
    dirdict = readJson()

    y_true = []
    y_test = []
    predslist = []

    for i in range(len(dirdict)):
        folder_name = test_subdirectory + "/" + dirdict['{}'.format(i)] +"/"
        y_true += [i for j in range(len([name for name in os.listdir(folder_name)]))]

        preds = testModelFunction(model_type=model_type, folder_path=folder_name, model_path=os.path.abspath(model_file))
        predslist += [value for key, value in preds.items()]

    for j in range(len(predslist)):
        for i in range(len(dirdict)):
            if (predslist[j][0][0] == dirdict['{}'.format(i)]):
                y_test += [i]

    cm = confusion_matrix(y_true, y_test)

    f, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot = True, linewidths=0.5, linecolor='red', fmt=".0f", ax=ax)
    plt.xlabel("pred")
    plt.ylabel("validation")
    plt.show()
