## Imports
from tkinter import *
from tkinter.filedialog import askopenfilename
import time
import os

# Current path for the API needs to be updated if the script is run on another device
path = "C:/Users/DL/Documents/code/GitHub/projet_test/ImageAI/"
os.chdir(path)

from PIL import Image, ImageTk
import keyboard
import shutil

###
from core import trainModelFunction, testModelFunction, readJson, getModelType

## Variables globales
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
files = []
model_type = getModelType()
dataset_path = path + "dataset/"

## Command functions
def chooseFiles():
    '''Open Windows Files Explorer and store the selected files into the list files'''
    FILETYPES = [("Images PNG",".png"),("Images JPG",".jpg")]
    files_path = askopenfilename(title = "Select a file ...", filetypes=FILETYPES)
    files.append(files_path)
    text = Label(frame, text=os.path.basename(files_path))
    text.pack()

def loadFiles():
    '''Store the selected images into the test folder in the dataset folder'''
    test_path = path + "dataset/test/"
    for file in files :
        try :
            shutil.move(file,test_path)
        except :
            pass
    text = Label(frame, text='Files successfully loaded!')
    text.pack()

def nextImage(number_image):
    img = Image.open(files[number_image])
    tk_img = ImageTk.PhotoImage(img)
    label_image = Label(frame_prediction, image=tk_img)
    label_image.image = tk_img
    label_image.pack()
    if number_image == len(files)-1:
        number_image=0
    else:
        number_image+=1
    return label_image

def openWindowTraining(model_file):
    bool = False

    newWindow = Toplevel(root)
    newWindow.title('Training Window')
    newWindow.geometry('{}x{}'.format(WINDOW_WIDTH, WINDOW_HEIGHT))

    def switch():
        trainModelFunction(model = model_type, dataset_directory = dataset_path, json_subdirectory = path, train_subdirectory = None, test_subdirectory = None, num_experiments=1, continue_from_model = model_file)

    button_train = Button(newWindow, text='Retrain the model with all training data', command=switch)
    button_train.pack(side=LEFT, padx=100, pady=100)

    button_quit = Button(newWindow, text='Exit', command=lambda:[root.destroy()])
    button_quit.pack(side=RIGHT, padx=100, pady=100)


def openWindowPrediction(photo):
    photo_path = path + "dataset/test/" + photo

    newWindow = Toplevel(root)
    newWindow.title('Predict Window')
    newWindow.geometry('{}x{}'.format(WINDOW_WIDTH, WINDOW_HEIGHT))

    frame_prediction = Frame(newWindow)
    frame_prediction.pack(side=LEFT)

    img = Image.open(photo_path)
    tk_img = ImageTk.PhotoImage(img)
    label_image = Label(frame_prediction, image=tk_img)
    label_image.image = tk_img
    label_image.pack()

    OptionDict = readJson()
    OptionList = [value for key, value in OptionDict.items()]

    variable = StringVar(newWindow)
    variable.set(OptionList[0])  # Changer pour qu'il donne la valeur de la prédiction de l'algo
    opt = OptionMenu(newWindow, variable, *OptionList)
    opt.pack(expand=YES)

    def user_valid(photo):
        '''When the barley growth stage is given by the user, we move the picture into the train>"barley growth stage" folder. For example : if the stage is "Levee", we move the picture from the test folder into train>Levee.'''
        img_path = path + "dataset/test/" + photo
        validation = variable.get()
        train_path = path + "dataset/train/" + validation
        shutil.move(img_path,train_path)

    valid_button = Button(newWindow, text='Valid', command=lambda:[user_valid(photo), newWindow.destroy()])
    valid_button.pack(expand=YES)

def user_not_valid():
    ''''''

def displayImage(image, window):
    '''Display an image on the left side of a window'''
    img = Image.open(image)
    canvas = Canvas(window, height=300, width=300)
    img = img.resize((300, 300), Image.BICUBIC)
    photo = ImageTk.PhotoImage(img)
    item = canvas.create_image(0,0, image = photo)
    canvas.pack()

def predictFiles():
    '''On utilise l'algorithme pour prédire la classe d'évolution de l'orge des photos sélectionnées par choose_files. Le résultat dénommé 'preds' est un dictionnaire de la forme {'Nom_image' : ['prédiction', probabilité]}'''
    for file in os.listdir():
        if file.endswith(".h5"):
            model_file = file
    test_path = path + "dataset/test/"
    preds = testModelFunction(model_type=model_type, folder_path=test_path, model_path=os.path.abspath(model_file))
    for photo in preds.keys():
        openWindowPrediction(photo)

    # Les images sont maintenant dans leurs folders respectifs dans train. On va maintenant créer une nouvelle fenêtre pour pouvoir entraîner le modèle sur toutes les données présentes dans le train

    openWindowTraining(model_file)

    """
    for file in os.listdir():
        if file.endswith(".h5"):
            model_file = file
    dataset_path = path + "dataset/"
    test_path = path + "dataset/test/"
    preds = testModelFunction(model_type=model_type, folder_path=test_path, model_path=os.path.abspath(model_file))
    return(preds)
    newWindow = Toplevel(root)
    newWindow.title('Predict Window')
    newWindow.geometry('{}x{}'.format(WINDOW_WIDTH,WINDOW_HEIGHT))
    img_full_path=[]
    for i in range(len(list(preds))):
        img_full_path.append(test_path+list(preds)[i])

    os.chdir(test_path)
    canvas = Canvas(newWindow, width = 300, height = 300)
    canvas.pack()
    img = ImageTk.PhotoImage(Image.open("C:/Users/geyma/Pictures/Fonds d'écran/Fond d'écran 1.jpg"))
    canvas.create_image(20, 20, anchor=NW, image=img)

    test = Label(newWindow,text='test')
    test.pack()


    button5 = Button(newWindow, text='Valid', command=lambda:[user_valid(preds, list(preds)[0]), displayImage(test_path+list(preds)[1],newWindow),button5.destroy()])
    button5.pack(side=RIGHT)

    button6 = Button(newWindow, text='Valid', command=lambda:[user_valid(preds, list(preds)[1]), displayImage(test_path+list(preds)[2]),newWindow])
    button6.pack(side=RIGHT)
    button6.destroy()

    button7 = Button(newWindow, text='Valid', command=user_valid(preds, list(preds)[2]))
    button7.pack(side=RIGHT)
    button7.destroy()


    for label_img in preds.keys():
        im = Image.open(label_img)
        image = ImageTk.PhotoImage(im, master=newWindow)
        dessin = tk.Canvas(fen, width = im.size[0], height = im.size[1])
        logo1 = dessin.create_image(0,0, anchor = tk.NW, image = logo)
        dessin.grid()

    #On demande à l'utilisateur de vérifier les prédictions
    for label_img in preds.keys():
        text = Label(root, text=label_img)
        text.pack()
        text = Label(root, text=preds[label_img])
        text.pack()
        button4 = Button(root, text='Valid', command=user_valid(preds, label_img))
        button4.pack()
        button5 = Button(root, text='Not Valid')
        button5.pack()
    #On stocke le modèle que l'on vient d'utiliser dans les archives
    shutil.move(model_file, 'model_archives')
    """


## Root custom window
root = Tk()
root.iconbitmap(path + "logo.ico")
root.title('Import Window')
root.geometry('{}x{}'.format(WINDOW_WIDTH,WINDOW_HEIGHT))
root.state("zoomed")

## Structure
frame = Frame(root)
frame.pack(expand=YES)

button1 = Button(frame, text='Choose files (.png)', command=chooseFiles)
button1.pack()

button2 = Button(root, text='Load files', command=lambda:[loadFiles(),predictFiles()])
button2.pack(expand=YES)

"""
button3 = Button(root, text='Predict', command=lambda:[predict_images,openWindow])
button3.pack(expand=YES)


adhar = Label(
    frame1,
    text='Upload Barley Image (png)'
    )
adhar.grid(row=0, column=0)

adharbtn = Button(
    frame1,
    text ='Choose File',
    command = lambda:open_file()
    )
adharbtn.grid(row=0,column=1)

frame1.grid()

upld = Button(
    frame2,
    text='Upload Files',
    command=uploadFiles
    )
upld.grid(row=1, column=0)

frame2.grid()
"""
root.mainloop()
