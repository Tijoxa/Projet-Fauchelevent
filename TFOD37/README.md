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
