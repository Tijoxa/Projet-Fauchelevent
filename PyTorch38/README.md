### PyTorch38

L'algorithme utilisant PyTorch38 s'inspire très largement des travaux du Kaggle "Getting started with object detection with pytroch : https://www.kaggle.com/aryaprince/getting-started-with-object-detection-with-pytorch. Il tourne sous Python 3.8.10.

Deux scripts sont présents dans ce dossier :
- "former main.py" : algorithme qui permet de détecter une classe sur une image à l'aide de bounding boxes
- "main pytorch.py" : algorithme qui permet de détecter une ou plusieurs classes sur une image à l'aide de bounding boxes labellisées

### Steps

<b>Step 1.</b> Installer les dependencies suivantes dans un environnement virtuel
<pre>
pip install numpy
pip install pandas
pip install opencv-python  # cv2
pip install pillow  # PIL
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
</pre>
<b>Step 1.bis</b> Pour utiliser le GPU lors de l'entraînement, installer CUDA 11.1 (version compatible avec cette version de Python et de Pytorch)

<b>Step 2.</b> Labellisation
La labellisation pour le script "former main.py" est enregistrée au format .csv, de la même façon que dans le Notebook Kaggle https://www.kaggle.com/aryaprince/getting-started-with-object-detection-with-pytorch

La labellisation pour le script "main pytorch.py" se fait de la même façon que pour la partie TFOD37

<b>Step 3.</b> Run le script Python en prenant en compte les commentaires
