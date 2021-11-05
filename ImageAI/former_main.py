path = "C:/Users/DL/Documents/code/GitHub/ImageAI/"

import sys
sys.path.append(path)

## class_weigth_custom
import os

dirlist = [item for item in os.listdir(path + "dataset/train/") if os.path.isdir(os.path.join(path + "dataset/train/", item))]

lengthLabel = [1.0 for i in range(len(dirlist))]

for i in range(len(dirlist)):
    folderName = path + "dataset/train/" + dirlist[i]+"/"
    lengthLabel[i] = len([name for name in os.listdir(folderName)])

maxLengthLabel = max(lengthLabel)

class_weight_custom = {i: maxLengthLabel/lengthLabel[i] for i in range(len(dirlist))}  # the json file provides the mapping between the folder name and the number of its class

## Train MobileNetV2
from imageaicustom import ClassificationModelTrainer

model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsMobileNetV2()
model_trainer.setDataDirectory(path + "dataset", json_subdirectory = path, test_subdirectory = path + "dataset/validation")

##
model_trainer.trainModel(num_objects=4, num_experiments=1, enhance_data=True, batch_size=4, training_image_size=1024, class_weight_custom=class_weight_custom)

## Train DenseNet121
from imageaicustom import ClassificationModelTrainer

model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsDenseNet121()
model_trainer.setDataDirectory(path + "dataset", json_subdirectory = path, test_subdirectory = path + "dataset/validation")

##
model_trainer.trainModel(num_objects=4, num_experiments=1, enhance_data=True, batch_size=2, training_image_size=1024, class_weight_custom=class_weight_custom)

## Train ResNet50
from imageaicustom import ClassificationModelTrainer

model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsResNet50()
model_trainer.setDataDirectory(path + "dataset", json_subdirectory = path, test_subdirectory = path + "dataset/validation")

##
model_trainer.trainModel(num_objects=4, num_experiments=1, enhance_data=True, batch_size=4, training_image_size=256, class_weight_custom=class_weight_custom)

## Train EfficientNetB7 - doesn't work
from imageaicustom import ClassificationModelTrainer

model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsEfficientNetB7()
model_trainer.setDataDirectory(path + "dataset", json_subdirectory = path, test_subdirectory = path + "dataset/validation")

##
model_trainer.trainModel(num_objects=4, num_experiments=5, enhance_data=True, batch_size=4, training_image_size=128, class_weight_custom=class_weight_custom)

## Test
from imageaicustom import CustomImageClassification

prediction = CustomImageClassification()
prediction.setModelTypeAsDenseNet121()
prediction.setModelPath(path + "dataset/models/model_ex-013_acc-0.510417.h5")
prediction.setJsonPath(path +"model_class.json")
prediction.loadModel(num_objects=4)

##
file_name = "0eee92ea416f01ff18578463a9ba9014dca90e8c33dde3f1f59592d52d082cab.png"

predictions, probabilities = prediction.classifyImage(path + "dataset/test/" + file_name, result_count=4)

print(predictions, probabilities)
