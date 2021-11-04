import os

# Current path for the API needs to be updated if the script is run on another device
path = "C:/Users/DL/Documents/Projet-Fauchelevent-Purjack-patch-1/ImageAI/"
os.chdir(path)

import sys
sys.path.append(path)

import json
from random import sample
import shutil

model = "mobilenetv2"  # mobilenetv2 / densenet121 / resnet50 / efficientnetb7

def getModelType():
    return model


test_subdirectory = path + "dataset/validation"
train_subdirectory = path + "dataset/train"

def readJson():
    with open(os.path.join(path, "model_class.json"), "r") as json_file:
        dirdict = json.load(json_file)
        json_file.close()
    return dirdict

## class_weigth_custom
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

def createValidationFolders():
    try :
        os.mkdir(path + "dataset/validation")
    except :
        pass

    L = os.listdir(path + "dataset/train")

    for label_name in L:
        try :
            os.mkdir(path + "dataset/validation/" + label_name)
        except :
            pass

        pourc = int(0.2*len(os.listdir(path + "dataset/train/" + label_name)))
        files = sample(os.listdir(path + "dataset/train/" + label_name), pourc)

        for file in files :
            shutil.move(path + "dataset/train/" + label_name + "/" + file, path + "dataset/validation/" + label_name)

def removeValidationFolders():
    L = os.listdir(path + "dataset/validation")

    for label_name in L:
        for file in os.listdir(path + "dataset/validation/" + label_name):
            shutil.move(path + "dataset/validation/" + label_name + "/" + file, path + "dataset/train/" + label_name)

    shutil.rmtree(path + "dataset/validation/")

def trainModelFunction(model, dataset_directory = path + "dataset", json_subdirectory = path, train_subdirectory = None, test_subdirectory = None, num_experiments = 50, continue_from_model = None):

    dirdict = readJson()

    model_trainer = ClassificationModelTrainer()

    if (model == "mobilenetv2"):
        model_trainer.setModelTypeAsMobileNetV2()
    elif (model == "densenet121"):
        model_trainer.setModelTypeAsDenseNet121()
    elif (model == "resnet50"):
        model_trainer.setModelTypeAsResNet50()
    elif (model == "efficientnetb7"):
        model_trainer.setModelTypeAsEfficientNetB7()

    model_trainer.setDataDirectory(dataset_directory, json_subdirectory=json_subdirectory, test_subdirectory = path + "dataset/validation", train_subdirectory = path + "dataset/train")
    model_trainer.trainModel(num_objects=len(dirdict), num_experiments=num_experiments, enhance_data=True, batch_size=4, training_image_size=1024, class_weight_custom=class_weight_custom, continue_from_model=continue_from_model)

    removeValidationFolders()

# trainModelFunction(test_subdirectory = test_subdirectory, train_subdirectory = train_subdirectory)

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

# print(testModelFunction("mobilenetv2"))

'''
## Confusion matrix
from tensorflow.math import confusion_matrix

def confusionMatrix(test_subdirectory):

    dirdict = readJson()

    y_true = []
    y_test=[]

    for i in range(len(dirdict)):
        folder_name = test_subdirectory + "/" + dirdict['{}'.format(i)] +"/"
        y_true += [i for j in range(len([name for name in os.listdir(folder_name)]))]
        name_list = [item for item in os.listdir(folder_name)]

        for file_name in name_list:
            preds = testModelFunction(file_path = folder_name + file_name)
            predictions, probabilities = preds[file_name]
            for i in range(len(dirdict)):
                if (predictions[0] == dirdict['{}'.format(i)]):
                    y_test += [i]

    print(confusion_matrix(y_true, y_test), y_test)

confusionMatrix()
'''
