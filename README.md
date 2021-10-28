# Projet-Fauchelevent

<img src="https://le-cdn.website-editor.net/22136137722b4fd9bef4eaa9defe6148/dms3rep/multi/opt/logo_farmbot_fauchelevent-596w.png">

Le projet Fauchelevent est un projet écologique mis en place par l'entreprise Metigate. L'objectif de ce projet est de déterminer l'impact du climat sur la production agricole via une collecte de données agronomiques. Pour y parvenir, Metigate nous a demandé de concevoir un algorithme permettant de déterminer 4 stades de croissance de l'orge à partir de photos prises par un robot potager Farmbot.

## La structure du dépôt GitHub

Ce dépôt GitHub recense à la fois le livrable final (ImageAI) mais aussi les pistes explorées puis abandonnées (TFOD et PyTorch). Nous avons dédié une partie explicative dans ce README pour chaque dossier présent à la racine du projet. Nous y avons explicité les intallations nécessaires, le fonctionnement de l'algorithme et les erreurs rencontrées. Nous avons aussi tenu à expliquer pourquoi nous avons abandonné les pistes TFOD et PyTorch pour une meilleure compréhension de notre démarche au cours des 7 semaines qui ont rythmé ce projet.

## ImageAI
## TFOD37

L'algorithme utilisant TFOD37 s'inspire très largement des travaux de Nicholas Renotte, que l'on peut retrouver à l'adresse : https://github.com/nicknochnack/TFODCourse. Il existe également une vidéo explicative (environ 5h) : https://youtu.be/yqkISICHH-U?list=RDCMUCHXa4OpASJEwrHrLeIzw7Yg.

Cet algorithme se divise en deux jupyter notebooks :
- 1 : Labellisation
- 2 : Installation de TFOD, training et testing

### Steps

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
<b>Step 5.</b> Labelliser les images en utilisant le Notebook <a href="https://github.com/Tijoxa/Projet-Fauchelevent/blob/main/TFOD37/1.%20Image%20Collection.ipynb">1. Image Collection.ipynb</a> - s'assurer que l'on a bien changé mis l'environnement virtuel en kernel comme montré sur la photo ci-dessous
<img src="https://i.imgur.com/8yac6Xl.png"> 
<br/>
<b>Step 6.</b> Diviser manuellement les images labellisées en deux dossiers train et test. Manually divide collected images into two folders train and test. Alors, tous les dossiers et annotations doivent maintenant être répartis entre les deux dossiers suivants. <br/>
\TFODCourse\Tensorflow\workspace\images\train<br />
\TFODCourse\Tensorflow\workspace\images\test
<br/><br/>
<b>Step 7.</b> Commencer le processus de training en ouvrant <a href="https://github.com/Tijoxa/Projet-Fauchelevent/blob/main/TFOD37/2.%20Training%20and%20Detection.ipynb">2. Training and Detection.ipynb</a>, ce notebook vous guidera dans l'installation de Tensorflow Object Detection, la réalisation de détections, la sauvegarde et l'exportation du modèle. 
<br /><br/>
<b>Step 8.</b> Au cours de ce processus, le Notebook installera Tensorflow Object Detection. Vous devriez idéalement recevoir une notification indiquant que l'API a été installée avec succès à l'étape 8 avec la dernière ligne indiquant OK.  
<img src="https://i.imgur.com/FSQFo16.png">
Si ce n'est pas le cas, résoudre les erreurs d'installation en se référant au <a href="https://github.com/Tijoxa/Projet-Fauchelevent/blob/main/TFOD37/Error%20Guide.md">Guide des erreurs.md</a> de ce dossier.
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

### Abandon de la piste TFOD

Nous avons décidé d'abandonner l'utilisation de TFOD car nous avons décidé d'abandonner la piste de détection d'images.En grande partie, cela est dû à notre jeu de données et le travail colossal que représente la labellisation.
Deux choix s'offraient à nous, chacun avec ses problèmes :
- Utiliser le dataset de la compétition Global Wheat Detection pré-labellisées : La labellisation de ce dataset consiste à placer des bounding boxes autour des épis sans les classifier. Cela est problématique, car il faut catégoriser chaque bounding boxes qui peut représenter les étapes 3 ou 4 des stades de croissance. Or cela représente plus de 30 bounding boxes pour 2000 images (donc plus de 60000 classifications à faire). De plus, il faut trouver des images des stades 1 et 2 de croissance et les labelliser avec labelImg, or cela est très compliqué il y a souvent une grande quantité de pousses/tallages sur une seule image, ce qui rend la labellisation très longue et délicate.
- Labelliser à la main des photos prises sur Internet : Comme expliqué plus haut (et comme on peut le voir sur l'image ci-dessous), il y a souvent une grande quantité de pousses/tallages/épis sur une seule image, ce qui rend la labellisation très longue et délicate.
<img src="https://agriculture.gouv.fr/sites/minagri/files/styles/affichage_pleine-page_790x435/public/epis_de_ble.jpg?itok=CvMknqge"> 
Pour résoudre le problème de la délicatesse de labelliser chaque plant individuellement, nous avons décidé de labelliser en grands groupes de stades d'évolution. Or, dans une même image, il était très rare que deux stades différents de croissance se présentent. Pour la plupart des images, cela revenait donc simplement à labelliser en plaçant une bounding box englobant toute l'image et en la classifiant selon le stade de croissance (souvent unique) que l'on pouvait voir. Nous avons donc simplifié le problème, en le considérant comme un simple problème de classification. Pour le résoudre, nous avons préféré utiliser l'outil ImageAI, très léger et facile à comprendre.

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
