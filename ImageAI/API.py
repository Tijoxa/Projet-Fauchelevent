## Imports
from tkinter import *
from tkinter.filedialog import askopenfilename
import time
import os
from PIL import Image, ImageTk

# Current path for the API needs to be updated in the main_for_API.py script if run on another device
from main_for_API import getPath
path = getPath()
os.chdir(path)

import shutil
import json
from main_for_API import testModelFunction, readJson

## Variables globales
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
files = []
model_type = "mobilenetv2"

## Command functions

def chooseFiles():
    '''Open Windows Files Explorer and store the selected files into the list files'''
    FILETYPES = [("Images PNG", ".png"), ("Images JPG", ".jpg")]
    files_path = askopenfilename(title = "Select a file ...", filetypes = FILETYPES)
    files.append(files_path)
    text = Label(frame, text = os.path.basename(files_path))
    text.pack()
    '''
    AFTER edit l.3 -> from tkinter.filedialog import askopenfilenames
    files_path = askopenfilenames(title = "Select a file ...", filetypes = FILETYPES)
    for i in range(len(files_path)):
        files.append(files_path[i])
        text = Label(frame, text = os.path.basename(files_path[i]))
        text.pack()
    '''

def loadFiles():
    '''Store the selected images into the test folder in the dataset folder'''
    test_path = path + "dataset/test/"
    for file in files :
        shutil.move(file, test_path)
    text = Label(frame, text = 'All files loaded!')
    text.pack()
    
def nextImage(number_image):
    img = Image.open(files[number_image])
    tk_img = ImageTk.PhotoImage(img)
    label_image = Label(frame_prediction, image = tk_img)
    label_image.image = tk_img
    label_image.pack()
    if number_image == len(files)-1:
        number_image = 0
    else:
        number_image += 1
    return label_image

def openWindowPrediction(photo):
    test_path = path + "dataset/test/"
    files_abspath = []
    for file in files :
        files_abspath.append(test_path + os.path.basename(file))

    newWindow = Toplevel(root)
    newWindow.title('Predict Window')
    #newWindow.geometry('{}x{}'.format(WINDOW_WIDTH, WINDOW_HEIGHT))
    newWindow.geometry('1920x1010')

    frame_prediction = Frame(newWindow)
    frame_prediction.pack(side = LEFT)

    number_image = 0

    img = Image.open(files_abspath[number_image])
    tk_img = ImageTk.PhotoImage(img)
    label_image = Label(frame_prediction, image = tk_img)
    label_image.image = tk_img
    label_image.pack()

    number_image += 1

    button_next = Button(newWindow, text = 'Image suivante', command = lambda:[label_image.destroy(), nextImage(number_image)])  # Rajouter la commande
    button_next.pack()

    # OptionList = ["Levee", "Tallage", "Epi", "Moisson"]  # Prendre depuis l'autre script :
    OptionDict = readJson()
    OptionList = [value for key, value in OptionDict.items()]
    
    variable = StringVar(newWindow)
    variable.set(OptionList[0])  # Changer pour qu'il donne la valeur de la prédiction de l'algo
    opt = OptionMenu(newWindow, variable, *OptionList)
    opt.pack(expand = YES)

    valid_button = Button(newWindow, text='Valid', command = lambda:[user_valid(), newWindow.destroy()])
    valid_button.pack(expand = YES)


def user_valid(preds, label_img):
    '''If prediction is correct, we move the image into the correspondant barley growth stage into the train folder'''
    img_path = path + "dataset/test/" + label_img
    prediction = preds[label_img][0][0]
    train_path = path + "dataset/train/" + prediction
    shutil.move(img_path, train_path)

def user_not_valid():
    ''''''

def displayImage(image, window):
    '''Display an image on the left side of a window'''
    img = Image.open(image)
    canvas = Canvas(window, height = 300, width = 300)
    img = img.resize((300, 300), Image.BICUBIC)
    photo = ImageTk.PhotoImage(img)
    item = canvas.create_image(0, 0, image = photo)
    canvas.pack()

def predictFiles():
    '''On utilise l'algorithme pour prédire la classe d'évolution de l'orge des photos sélectionnées par choose_files. Le résultat dénommé 'preds' est un dictionnaire de la forme {'Nom_image' : ['prédiction', probabilité]}'''
    for file in os.listdir():
        if file.endswith(".h5"):
            model_file = file
    print(os.path.abspath(model_file))
    test_path = path + "dataset/test/"
    os.chdir(test_path)
    preds = testModelFunction(model_type = model_type, folder_path = test_path, model_path = model_file)
    for photo in preds.keys():
        openWindowPrediction(photo)

    """
    for file in os.listdir():
        if file.endswith(".h5"):
            model_file = file
    dataset_path = path + "dataset/"
    test_path = path + "dataset/test/"
    preds = testModelFunction(model_type = model_type, folder_path = test_path, model_path = os.path.abspath(model_file))
    return(preds)
    newWindow = Toplevel(root)
    newWindow.title('Predict Window')
    newWindow.geometry('{}x{}'.format(WINDOW_WIDTH, WINDOW_HEIGHT))
    img_full_path = []
    for i in range(len(list(preds))):
        img_full_path.append(test_path + list(preds)[i])

    os.chdir(test_path)
    canvas = Canvas(newWindow, width = 300, height = 300)
    canvas.pack()
    img = ImageTk.PhotoImage(Image.open("C:/Users/geyma/Pictures/Fonds d'écran/Fond d'écran 1.jpg"))
    canvas.create_image(20, 20, anchor = NW, image = img)

    test = Label(newWindow,text='test')
    test.pack()

    button5 = Button(newWindow, text = 'Valid', command = lambda:[user_valid(preds, list(preds)[0]), displayImage(test_path + list(preds)[1], newWindow), button5.destroy()])
    button5.pack(side = RIGHT)

    button6 = Button(newWindow, text = 'Valid', command = lambda:[user_valid(preds, list(preds)[1]), displayImage(test_path + list(preds)[2]), newWindow])
    button6.pack(side = RIGHT)
    button6.destroy()

    button7 = Button(newWindow, text = 'Valid', command = user_valid(preds, list(preds)[2]))
    button7.pack(side = RIGHT)
    button7.destroy()


    for label_img in preds.keys():
        im = Image.open(label_img)
        image = ImageTk.PhotoImage(im, master = newWindow)
        dessin = tk.Canvas(fen, width = im.size[0], height = im.size[1])
        logo1 = dessin.create_image(0,0, anchor = tk.NW, image = logo)
        dessin.grid()

    #On demande à l'utilisateur de vérifier les prédictions
    for label_img in preds.keys():
        text = Label(root, text = label_img)
        text.pack()
        text = Label(root, text = preds[label_img])
        text.pack()
        button4 = Button(root, text = 'Valid', command = user_valid(preds, label_img))
        button4.pack()
        button5 = Button(root, text = 'Not Valid')
        button5.pack()
    #On stocke le modèle que l'on vient d'utiliser dans les archives
    shutil.move(model_file, 'model_archives')
    """


## Root custom window
root = Tk()
root.title('Import Window')
root.geometry('{}x{}'.format(WINDOW_WIDTH, WINDOW_HEIGHT))

## Structure
frame = Frame(root)
frame.pack(expand = YES)

text = Label(frame, text = 'Load files (.png)')
text.pack()

button1 = Button(frame, text = 'Choose files', command = chooseFiles)
button1.pack()

button2 = Button(root, text = 'Load files', command = lambda:[loadFiles(), predictFiles()])
button2.pack(expand = YES)

"""
button3 = Button(root, text = 'Predict', command=lambda:[predict_images, openWindow])
button3.pack(expand = YES)


adhar = Label(
    frame1,
    text = 'Upload Barley Image (png)'
    )
adhar.grid(row = 0, column = 0)

adharbtn = Button(
    frame1,
    text = 'Choose File',
    command = lambda:open_file()
    )
adharbtn.grid(row = 0, column = 1)

frame1.grid()

upld = Button(
    frame2,
    text = 'Upload Files',
    command = uploadFiles
    )
upld.grid(row = 1, column = 0)

frame2.grid()
"""
root.mainloop()
