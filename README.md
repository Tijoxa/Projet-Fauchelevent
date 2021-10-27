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

L'algorithme utilisant TFOD37 s'inspire très largement des travaux de Nicholas Renotte, que l'on peut retrouver à l'adresse : https://github.com/nicknochnack/TFODCourse. Il existe également une vidéo explicative (environ 5h) : https://youtu.be/yqkISICHH-U?list=RDCMUCHXa4OpASJEwrHrLeIzw7Yg.

Cet algorithme se divise en deux jupyter notebooks :
- 1 : Labellisation
- 2 : Installation de TFOD, training et testing

### Steps
<br />
<b>Step 1.</b> Cloner ce dépôt: https://github.com/nicknochnack/TFODCourse dans un dossier nommé TFODCourse
<br/><br/>
<b>Step 2.</b> Créer un nouvel environnement virtuel nommé tfod (pour tensorflow object detection) dans le dossier TFODCourse
<pre>
python -m venv tfod
</pre> 
<br/>
<b>Step 3.</b> Activer l'environnement virtuel tfod
<pre>
source tfod/bin/activate # Linux
.\tfod\Scripts\activate # Windows 
</pre>
<br/>
<b>Step 4.</b> Installer les dépendances et ajouter l'environnement virtuel au Kernel Python
<pre>
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=tfodj
</pre>
<br/>
<b>Step 5.</b> Labelliser les images en utilisant le Notebook <a href="https://github.com/nicknochnack/TFODCourse/blob/main/1.%20Image%20Collection.ipynb">1. Image Collection.ipynb</a> - s'assurer que l'on a bien changé mis l'environnement virtuel en kernel comme montré sur la photo ci-dessous
<img src="https://i.imgur.com/8yac6Xl.png"> 
<br/>
<b>Step 6.</b> Diviser manuellement les images labellisées en deux dossiers train et test. Manually divide collected images into two folders train and test. Alors, tous les dossiers et annotations doivent maintenant être répartis entre les deux dossiers suivants. <br/>
\TFODCourse\Tensorflow\workspace\images\train<br />
\TFODCourse\Tensorflow\workspace\images\test
<br/><br/>
<b>Step 7.</b> Commencer le processus de training en ouvrant <a href="https://github.com/nicknochnack/TFODCourse/blob/main/2.%20Training%20and%20Detection.ipynb">2. Training and Detection.ipynb</a>, ce notebook vous guidera dans l'installation de Tensorflow Object Detection, la réalisation de détections, la sauvegarde et l'exportation du modèle. 
<br /><br/>
<b>Step 8.</b> Au cours de ce processus, le Notebook installera Tensorflow Object Detection. Vous devriez idéalement recevoir une notification indiquant que l'API a été installée avec succès à l'étape 8 avec la dernière ligne indiquant OK.  
<img src="https://i.imgur.com/FSQFo16.png">
Si ce n'est pas le cas, résoudre les erreurs d'installation en se référant au <a href="https://github.com/nicknochnack/TFODCourse/blob/main/README.md">Guide des erreurs.md</a> de ce dossier.
<br /> <br/>
<b>Step 9.</b> Une fois que vous êtes arrivé à l'étape 6. Entraîner le modèle à l'intérieur du carnet de notes, vous pouvez choisir d'entraîner le modèle à l'intérieur du carnet de notes. Nous avons cependant remarqué que l'entraînement à l'intérieur d'un terminal séparé sur une machine Windows permet d'afficher des mesures de perte en direct. 
<img src="https://i.imgur.com/K0wLO57.png"> 
<br />
<b>Step 10.</b> On peut éventuellement évaluer le modèle à l'intérieur de Tensorboard. Une fois que le modèle a été formé et que vous avez exécuté la commande d'évaluation à l'étape 7. Naviguer vers le dossier d'évaluation de votre modèle formé. e.g. 
<pre> cd Tensorlfow/workspace/models/my_ssd_mobnet/eval</pre> 
et ouvrir Tensorboard avec la commande suivante
<pre>tensorboard --logdir=. </pre>
Tensorboard sera accessible via votre navigateur et on pourra voir des mesures telles que la mAP (précision moyenne) et le Recall.
<br />

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
