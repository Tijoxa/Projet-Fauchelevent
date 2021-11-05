# ImageAI
## Avertissement
L'ensemble du README se veut le plus exhaustif possible. Nous avons écrit ce README de telle sorte que n'importe qu'elle personne puisse comprendre notre cheminement, les scripts python et comment modifier les scripts à sa guise pour faire tourner l'API chez soi. <b>Il est donc très important de tous lire !</b>
## Introduction
ImageAI est une bibliothèque Python de Computer Vision facile à utiliser qui permet aux développeurs d'intégrer facilement des fonctions d'intelligence artificielle de pointe dans leurs applications et systèmes nouveaux et existants. Elle est utilisée par des milliers de développeurs, étudiants, chercheurs, tuteurs et experts dans des entreprises du monde entier. 

Nous avons choisi cette bibliothèque pour la simplicité qu'elle confère par rapport à d'autres outils. ImageAI se résume en peu de lignes de codes car elle utilise des fonctions de Tensorflow (bibliothèque beaucoup plus dense).

## But du projet
L'explication du projet Fauchelevent a déjà été expliqué dans le README général. Notre travail consiste à fournir un modèle permettant de prédire le stade d'évolution de l'orge à partir de photos prises par un robot Farmbot. Pour ce faire, nous avons utilisé le dataset Global Wheat pour pré-entraîner plusieurs modèles à la détection de céréales. Le fait de pré-entraîner sur des images de blé ne pose pas vraiment de problème car le blé et l'orge pousse de façon presque identique. Le livrable final est une API permettant de prédire le stade d'évolution de nouvelles photos prises par le robot Farmbot grâce au modèle pré-entraîné sur Global Wheat Dataset. Elle permet également à l'utilisateur de vérifier ces prédictions et de les corriger si nécessaire. Une fois correctement labellisées, ces photos sont utilisées comme données d'entraînement. L'API permet donc de ré-entraîner le modèle avec l'ensemble des données (i.e. les anciennes photos du Farmbot + les nouvelles photos).

Cette API a pour but de permettre à l'utilisateur de centraliser les tâches de prédiction, de labellisation et d'entraînement. A chaque ré-entraînement, le modèle s'ajuste pour prédire de mieux en mieux les stades d'évolution de l'orge.

## Structure du dossier
Le structure du dossier Image AI est la suivante :
<pre>
> ImageAI  
  > dataset
  > logs
  > models
  > test
  > train
    > Epi
    > Levee
    > Moisson
    > Tallage
  API.py
  core.py
  imageaicustom.py
  logo.ico
  model_class.json
  README.md
  requirements.txt
</pre>

### API
Le script python principal se situe dans <u>API.py</u> : Lancer ce script permet de lancer l'API.

On trouve d'abord la bibliothèque d'imports :
- le module tkinter fournit des outils de création d'APIs (askopenfilenames permettant de sélectionner des fichiers dans l'explorateur de fichiers)
- le module time fournit différentes fonctions liées au temps
- le module os fournit une façon portable d'utiliser les fonctionnalités dépendantes du système d'exploitation
- le module glob permet de rechercher des chemins de style Unix selon certains motifs
- le module PIL (diminutif de Pillow) est une bibliothèque de traitement d'image
- le module keyboard permet de contrôler les actions du clavier
- le module random.sample est utilisé pour un échantillonage aléatoire
- le module shutil offre un certain nombre d'opérations de haut niveau sur les fichiers et les collections de fichiers
- on appelle les fonctions trainModelFunction, testModelFunction, getModelType, readJson, confusionMatrix du script core.py

Le path est le dossier courant dans lequel se situe l'ensemble du dossier ImageAI. Il doit donc se terminer par ".../ImageAI/". Pour faire tourner l'API chez soi, <b> IL FAUT CHANGER A LA MAIN LE NOM DU PATH DANS API.PY</b>, faute de quoi l'algorithme ne trouvera pas vos fichiers.

On définit par la suite des variables globales que l'on pourra appeler n'importe où dans le script. Elles servent au dimensionnement de fenêtres, au choix des images, à la définition du type de modèle utilisé parmi mobilenetv2, resnet50, densenet121, inceptionv3 ou efficientnetb7. 

Vient ensuite une liste de fonctions utilisées pour la plupart lorsque l'on appuie sur des boutons dans l'API. Nous ne nous attarderons pas sur l'explication de ces fonctions qui sont chacune déjà commentées dans le script API.py. Il est cependant important de noter que le nombre d'epochs lorsque l'on ré-entraîne l'algorithme peut être changé à la main en <b>éditant la variable num_experiments dans la fonction openWindowTraining</b>.

Nous créons donc notre fenêtre principale que nous customisons avec des dimensions, un logo (logo.ico) et un titre. Puis, nous créons un cadre pour placer correctement des boutons et du texte. Le root.mainloop() est simplement une méthode dans la fenêtre principale qui exécute ce que nous souhaitons exécuter dans une application (permet à Tkinter de commencer à exécuter l'application). Comme son nom l'indique, elle bouclera indéfiniment jusqu'à ce que l'utilisateur quitte la fenêtre ou attende tout événement de la part de l'utilisateur.

### Core

Le script core.py est appelé par le script API.py. Dans core.py, on retrouve notamment des fonctions relatives à l'entraînement du modèle, à la prédiction sur de nouvelles images et à l'affichage de la matrice de confusion.<b> IL FAUT CHANGER A LA MAIN LE NOM DU PATH DANS CORE.PY</b>. Il est également important de savoir que <b>pour changer le type de modèle utilisé, il faut éditer à la main la variable model_type</b>. L'ensemble des modèles disponibles supportés facilement par ImageAI sont mobilenetv2, resnet50, densenet121, inceptionv3 ou efficientnetb7.

L'ensemble des fonctions utilisées dans core.py sont également commentées dans le script.

### Imageaicustom

Sur le site d'ImageAI (http://www.imageai.org/), on peut trouver le gitHub relatif à la bibliothèque en cliquant sur "GitHub Repository" (https://github.com/OlafenwaMoses/ImageAI). Là, il faut aller dans le dossier ImageAI > imageai > Classification > Custom pour trouver le script __init.py__. Imageaicustom n'est rien d'autre que ce script __init.py__ que nous avons un peu modifié pour faire marcher certaines de nos fonctions dans core.py ou dans API.py.

Les fonctions sont également commentées dans ce script python.

### model_class.json

Ce fichier Json renseigne le nom des stades d'évolution de l'orge que nous voulons prédire. Il y en a 4 :
- Stade 1 = "Levee"
- Stade 2 = "Tallage"
- Stade 3 = "Epi"
- Stade 4 = "Moisson"

Dans le fichier Json, les stades d'évolution ne sont pas classés comme ci-dessous car celui-ci les classent par ordre alphanumérique :
- 1 : Epi
- 2 : Levee
- 3 : Moisson
- 4 : Tallage

<i>Note : Si l'utilisateur souhaite renommer les satdes d'évolution pour avoir un ordre plus logique dans le fichier Json, il doit également comprendre que les modèles pré-entraînés (sur le dataset Global Wheat) que nous fournissons sont liés au Json "mal ordonné". Pour pouvoir exécuter l'API, il conviendra alors de repartir sur des modèles vierges en initialisant l'argument continue_from_model à None dans la fonction switch de la fonction openWindowTraining dans le script API.py.</i>

Nous expliquerons un peu plus tard comment différencier ces stades de croissance pour labelliser correctement.

### Dataset

Ce dossier contient l'ensemble des images qui seront utiles à l'algorithme, les modèles et des fichiers ouvrables dans TensorBoard pour visualiser la performance du modèle. En particulier : 
- le dossier "train" contient l'ensemble des données d'entraînement
- le dossier "test" est vide au début puis servira de dossier de stockage des nouvelles images avant leur prédiction par le modèle
- le dossier "models" est un dossier de stockage du nouveau modèle après ré-entraînement de l'ancien modèle sur les anciennes et les nouvelles images. Ce dossier est vide au début
- le dossier "models_archives" n'existe pas au début, il est créé lors du premier ré-entraînement de l'algorithme dans l'API. C'est un dossier de stockage des anciens modèles.
- le dossier "logs" permet de stocker les dossiers logs de chaque occurence du ré-entraînement de l'algorithme. Dans ces dossiers logs, on retrouve notamment des fichiers V2 ouvrables sur TensorBoard.

## Installations nécessaires

On retrouve l'ensemble des installations nécessaire ci-dessous dans requirements.txt :

<pre>
Python 3.7.6 (with tkinter Tcl/Tk)
pip
pip install tensorflow==2.4.0
pip install tensorflow-gpu==2.4.0
pip install keras==2.4.3 numpy==1.19.3 pillow==7.0.0 scipy==1.4.1 h5py==2.10.0 matplotlib==3.3.2 opencv-python keras-resnet==0.2.0
pip install keyboard
pip install seaborn
</pre>

## Utilisation de l'API

Avant d'exécuter le script API.py, il convient de faire quelques manipulations. En particulier, il faut télécharger l'ensemble des modèles pré-entraînés sur Global Wheat à l'adresse https://mega.nz/file/KMR0zZrb#i2IoAQzYbnAXlaHHPB7qnMqjilWWX99H9A-n2HtYbZ4. Pour les plus curieux, on pourra également trouver les images du dataset Global Wheat Detection à l'adresse https://mega.nz/file/CEJTgQ4A#bfYrD2gzBqJyfbK9Byss981Bks2rBpZ6_lEEEt0a4HI. <i>Ces liens ne seront plus garantis à partir de 2022/12/31 (YYYY/MM/DD).</i>

Parmi les 3 modèles .h5 que l'on peut trouver à la première adresse, il faut en choisir un et le placer dans le dossier ImageAI (i.e dans le même dossier que les fichiers python). Ces trois modèles ont été pré-entraînés sur le dataset Global Wheat Detection qui est très largement déséquilibré. Malgré nos tentatives de rééquilibrage, ces modèles ne performent pas au maximum de leurs capacités (entre 0.62 et 0.73 et val_accuracy). Nous laissons donc le choix à l'utilisateur de prendre nos modèles pré-entraînés sur Global wheat ou bien de partir des modèles vierges.

<b>On peut enfin lancer le script Python !</b>

L'API s'ouvre sur une fenêtre avec les boutons "Choose files" et "Load Files". Le bouton Choose Files permet de parcourir l'explorateur de fichiers pour choisir les nouvelles images du robot Farmbot. Les formats PNG et JPG sont supportés pour les images. Une fois toutes les images choisies, on appuie sur "Load Files" qui va les transférer dans le dossier dataset > test.

Puis s'ouvrent une par une les fenêtres de labellisation. L'API renseigne la prédiction que fait le modèle (le fichier .h5 choisi et placé dans le dossier ImageAI) et autorise l'utilisateur à rectifier la prédiction si elle est fausse. Il peut pour cela utiliser le menu déroulant puis cliquer sur "Valid". Pour la labellisation, il convient de se renseigner sur la pousse d'orge. Pour l'équipe Fauchelevent, ces stades de croissance sont définies dans la slide 14 de la présentation PowerPoint présentes sur le Drive.

Une fois toutes les labellisations faites, les nouvelles images sont stockées dans le jeu d'entraînement et une nouvelle fenêtre s'affiche. Les :
- Segmenter le jeu de données en un jeu d'entraînement et de validation (Crée donc un dossier Validation au même endroit que le dossier train)
- Ré-entraîner le modèle (cela affiche à la fin la matrice de confusion)
- Une fois l'entraînement complété, on remet les images ayant servies à la validation dans le dossier train
- Puis on ferme le programme

A la toute fin, on peut évaluer les performances de l'algorithme avec l'outil TensorBoard (outil de visualisation fournit par Tensorflow qu'utilise ImageAI). Pour ce faire, on ouvre un interface de commande. Ensuite, il faut se placer dans le dossier logs correspondant et ouvrir Tensorboard :
<pre>
cd xxxxx\Projet-Fauchelevent\ImageAI\dataset\logs\xxxxx
python -m tensorboard.main --logdir=.
</pre>
Cela renseigne l'URL localhost qu'il faut ouvrir sur Internet pour avoir le Tensorboard.

## Pour aller plus loin (Pour l'équipe Fauchelevent notamment)

Pour aller plus loin, nous proposons de :
- Utiliser la date de photographie (i.e. la date de pousse) comme paramètre d'entrée du modèle pour augmenter ses performances
- Automatiser la prise de photos chez Sixmon
- Incliner le plan de photo (30°) avec la pièce Onshape
- Rendre l'API plus "user friendly"
- Customiser la data augmentation (dans imageaicustom.py) avec plus d'attention
