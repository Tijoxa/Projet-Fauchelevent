import os

os.chdir("C:/Users/Tijoxa/Desktop/code/project with ImageAI")

## Train
from imageai.Classification.Custom import ClassificationModelTrainer

model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsDenseNet121()
model_trainer.setDataDirectory(r"C:/Users/Tijoxa/Desktop/code/dataset Global Wheat", json_subdirectory = r"C:/Users/Tijoxa/Desktop/code/project with ImageAI")

##
model_trainer.trainModel(num_objects=4, num_experiments=2, enhance_data=True, batch_size=32, training_image_size=1024, show_network_summary=True)

## Test
from imageai.Classification.Custom import CustomImageClassification

prediction = CustomImageClassification()
prediction.setModelTypeAsDenseNet121()
prediction.setModelPath("C:/Users/Tijoxa/Desktop/code/project with ImageAI/models/")
prediction.setJsonPath("model_class.json")
prediction.loadModel(num_objects=4)

##
predictions, probabilities = prediction.classifyImage("C:/Users/Tijoxa/Desktop/code/dataset Global Wheat/test/0a243124ae3a8505916a5d4674510f6cf4a63619d0ccc2b3b9a36ca69d6adedc.png", result_count=1)

print(predictions, probabilities)