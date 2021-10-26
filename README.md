# Projet-Fauchelevent

Le projet Fauchelevent est un projet écologique mis en place par l'entreprise Metigate. L'objectif de ce projet est de déterminer l'impact du climat sur la production agricole via une collecte de données agronomiques. Pour y parvenir, Metigate nous a demandé de concevoir un algorithme permettant de déterminer 4 stades de croissance de l'orge à partir de photos prises par un robot potager Farmbot.

## La structure du dépôt gitHub

Nous avons d'abord abordé le problème avec les outils TensorFlow Object Detection (TFOD) et PyTorch. L'objectif était de faire de la détection d'images : l'algorithme serait capable d'identifier individuellement chaque plant grâce à des bounding boxes et de les classifier : pousse, tallage, épi ou moisson (les 4 stades de croissance). En pratique, cela a été difficile car la labellisation des images étaient trop chronophage étant donné que l'algorithme s'entraînait sur des images de champs. Nous avons donc réduit le problème à un simple problème de classification. Nous avons donc utilisé ImageAI.

Le dépôt gitHub est donc constitué du reste de nos pistes sur TFOD et PyTorch :
- TFOD37 (Version de Python 3.7)
- PyTorch38 (Version de Python 3.8)

Il est également constitué de notre algorithme final :
- ImageAI

## ImageAI
## TFOD37

L'algorithme utilisant TFOD37 s'inspire très largement des travaux de Nicholas Renotte, que l'on peut retrouver à l'adresse : https://github.com/nicknochnack/TFODCourse. Il existe également une vidéo explicative : https://youtu.be/yqkISICHH-U?list=RDCMUCHXa4OpASJEwrHrLeIzw7Yg.

Cet algorithme se divise en deux jupyter notebooks :
- 1 : Labellisation
- 2 : Installation de TFOD, training et testing

Il est grandement conseillé de réaliser les manipulations faites sur les notebooks dans un environnement virtuel, comme il est conseillé dans le README de Nicholas Renotte.

## PyTorch38

main.py est divisé en plusieurs sous-codes qui seront bientôt divisés en plusieurs scripts

librairies à installer :
- numpy
- pandas
- cv2 (opencv-python)
- pil (pillow)
- torch
- torchvision
- matplotlib

/root :
main.py
output.png
fasterrcnn_resnet50_fpn_best.pth   ( https://www.kaggle.com/mathurinache/fasterrcnn )
model_saved
train.csv   ( https://www.kaggle.com/c/global-wheat-detection/data )
sample_submission.csv   ( https://www.kaggle.com/c/global-wheat-detection/data )
/testflv
/train   ( https://www.kaggle.com/c/global-wheat-detection/data )

/root/testflv :
10 images 1024*1024

/root/train :
beaucoup d'images 1024*1024
