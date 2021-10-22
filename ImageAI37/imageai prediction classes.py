## Train
from imageai.Classification.Custom import ClassificationModelTrainer

model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsMobileNetV2()
model_trainer.setDataDirectory(r"C:/Users/DL/Documents/code/dataset Global Wheat", json_subdirectory = r"C:/Users/DL/Documents/code/project with ImageAI", train_subdirectory = "C:/Users/DL/Documents/code/dataset Global Wheat/train balanced a la zeub")

##
model_trainer.trainModel(num_objects=4, num_experiments=50, enhance_data=True, batch_size=4, training_image_size=1024)

## Test
from imageai.Classification.Custom import CustomImageClassification

prediction = CustomImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath("C:/Users/DL/Documents/code/dataset Global Wheat/models/model_ex-038_acc-0.781022.h5")
prediction.setJsonPath("C:/Users/DL/Documents/code/project with ImageAI/model_class.json")
prediction.loadModel(num_objects=4)

##
predictions, probabilities = prediction.classifyImage("C:/Users/DL/Documents/code/dataset Global Wheat/test/Levee/0aa3cb8cd58e8c7b55fd5ede8acebb0ae5fd3f072f8f70f88f3dd7c55a67f5a0.png", result_count=4)

print(predictions, probabilities)

# Résultat : dataset pas équilibré, donc prédit Epi tout le temps

## Data augmentation to balance dataset  tf.keras.preprocessing.image.ImageDataGenerator
import os

maxLengthLabel = 0

root='C:/Users/DL/Documents/code/dataset Global Wheat/train/'
dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]

for i in range(len(dirlist)):
    folderName = "C:/Users/DL/Documents/code/dataset Global Wheat/train/" + dirlist[i]+"/"
    maxLengthLabel = max(maxLengthLabel, len([name for name in os.listdir(folderName)]))

for i in range(len(dirlist)):
    folderName = "C:/Users/DL/Documents/code/dataset Global Wheat/train/" + dirlist[i]+"/"

    while (len([name for name in os.listdir(folderName)]) < maxLengthLabel):
        createRandomImage("truc")  # fonction à écrire

##
