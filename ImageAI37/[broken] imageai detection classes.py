## Train
from imageai.Detection.Custom import DetectionModelTrainer

model_trainer = DetectionModelTrainer()

model_trainer.setModelTypeAsYOLOv3()

model_trainer.setDataDirectory(r"C:/Users/Tijoxa/Desktop/code/maxime/Tensorflow Object Detection/TFODCourse/Tensorflow/workspace/images")

model_trainer.setTrainConfig(object_names_array=["Levée", "Moisson", "Tallage", "Epi"], num_experiments=100, batch_size=32)

model_trainer.trainModel()
