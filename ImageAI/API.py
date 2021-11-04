## Imports
from tkinter import *
from tkinter.filedialog import askopenfilenames
import time
import os
from PIL import Image, ImageTk
import keyboard

# Current path for the API needs to be updated if the script is run on another device
path = "C:/Users/DL/Documents/Projet-Fauchelevent-Purjack-patch-1/ImageAI/"
os.chdir(path)

import shutil
from core import trainModelFunction, testModelFunction, getModelType, readJson, createValidationFolders

## Global variables
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
files = []
model_type=getModelType()
dataset_path = path + "dataset/"

## Command functions

def chooseFiles():
    '''Open Windows Files Explorer and store the selected files into the list files'''
    FILETYPES = [("Images PNG",".png"),("Images JPG",".jpg")]
    files_path = askopenfilenames(title = "Select a file ...", filetypes=FILETYPES)
    for i in range(len(files_path)):
        files.append(files_path[i])
        text = Label(frame, text=os.path.basename(files_path[i]))
        text.pack()

def loadFiles():
    '''Store the selected images into the test folder in the dataset folder'''
    try :
        test_path = path + "dataset/test/"
        for file in files :
            shutil.move(file,test_path)
        text = Label(frame, text='Files successfully loaded!')
        text.pack()
    except :
        newWindow = Toplevel(root)

        def center_window(newWindow):
            eval = newWindow.nametowidget('.').eval
            eval_('tk::PlaceWindow %s center' % newWindow)

        center_window(newWindow)
        text = Label(newWindow, text = 'Could not load images...')
        text.pack(expand=YES)

        quit_button = Button(newWindow, text = 'Exit', command=root.destroy)
        quit_button.pack(expand=YES)

        newWindow.wait_window()


def openWindowTraining(model_file):
    '''Open a window for the training of the model. In this window, we can choose to retrain the model with all the training images (i.e. preivous training images + new images chosed in chooseFiles). We can also choose to exit the program.'''
    num_experiments = 200
    newWindow = Toplevel(root)
    newWindow.title('Training Window')
    newWindow.geometry('{}x{}'.format(WINDOW_WIDTH, WINDOW_HEIGHT))
    newWindow.state("zoomed")

    def switch():
        trainModelFunction(model = model_type, dataset_directory = dataset_path, json_subdirectory = path, train_subdirectory = None, test_subdirectory = None, num_experiments=num_experiments, continue_from_model = model_file)

    def split():
        createValidationFolders()

    button_split = Button(newWindow, text='Split data into train (80%) and validation (20%)', command=split)
    button_split.pack(side=LEFT, padx=100, pady=100)

    button_split = Button(newWindow, text='Train the model with {} for {} epochs'.format(getModelType(), num_experiments), command=switch)
    button_split.pack(side=LEFT, padx=100, pady=150)

    button_quit = Button(newWindow, text='Exit', command=lambda:[root.destroy()])
    button_quit.pack(side=RIGHT, padx=100, pady=100)


def openWindowPrediction(preds):
    '''Open a window for prediction. Into this window, we have a picture with its initial prediction from the model. The user needs to confirm or deny this prediction by using the correct stage of barley growth in the scrolling menu. When it's done, press Valid.'''

    newWindow = Toplevel(root)
    newWindow.title('Predict Window')
    newWindow.geometry('1920x1010')
    newWindow.state("zoomed")

    frame_prediction = Frame(newWindow)
    frame_prediction.pack(side=LEFT)

    img = Image.open(path + "dataset/test/" + os.listdir(path + "dataset/test/")[0])
    resized_img = img.resize((800, 800), Image.ANTIALIAS)
    tk_img = ImageTk.PhotoImage(resized_img)
    label_image = Label(frame_prediction, image=tk_img)
    label_image.image = tk_img
    label_image.pack(expand=YES)

    text_count = Label(frame_prediction, text='Image n°{}/{}'.format(list(preds.keys()).index(os.listdir(path + "dataset/test/")[0])+1,len(preds)))
    text_count.pack(expand=YES)

    text_pred = Label(newWindow, text="Initial prediction : {}".format(preds[os.listdir(path + "dataset/test/")[0]][0][0]) + " with {0:.2f}% level of confidence".format(float(preds[os.listdir(path + "dataset/test/")[0]][1][0])))
    text_pred.pack(expand=YES)

    OptionDict = readJson()
    OptionList = [value for key, value in OptionDict.items()]

    variable = StringVar(newWindow)
    variable.set(preds[os.listdir(path + "dataset/test/")[0]][0][0])  # Variable is initially set to the prediction made by the model
    opt = OptionMenu(newWindow, variable, *OptionList)
    opt.pack(expand=YES)

    def user_valid():
        '''When the barley growth stage is given by the user, we move the picture into the train>"barley growth stage" folder. For example : if the stage is "Levee", we move the picture from the test folder into train>Levee.'''
        validation = variable.get()
        train_path = path + "dataset/train/" + validation
        shutil.move(path + "dataset/test/" + os.listdir(path + "dataset/test/")[0],train_path)

    valid_button = Button(newWindow, text='Valid', command=lambda:[user_valid(), newWindow.destroy()])
    valid_button.pack(expand=YES)

    quit_button = Button(newWindow, text = 'Exit', command=root.destroy)
    quit_button.pack(expand=YES)

    return(newWindow)


def predictFiles():
    '''We use the algorithm to predict the barley growth stage of the pictures selected by choose_files. The result named 'preds' is a dictionary of the form {'Image_name' : ['prediction', probability]}'''
    for file in os.listdir():
        if file.endswith(".h5"):
            model_file = file
    test_path = path + "dataset/test/"
    preds = testModelFunction(model_type=model_type, folder_path=test_path, model_path=os.path.abspath(model_file))
    while len(os.listdir(path + "dataset/test/")):
        newWindow = openWindowPrediction(preds)
        newWindow.wait_window()

    # All images are now in their respective folders in the train. We will now create a new window to train the model on all the data present in the train
    openWindowTraining(model_file)


## Root custom window
root = Tk()
try :
    root.iconbitmap(path + "logo.ico")
except :
    pass
root.title('Import Window')
root.geometry('{}x{}'.format(WINDOW_WIDTH, WINDOW_HEIGHT))
root.state("zoomed")

## Structure
frame = Frame(root)
frame.pack(expand=YES)

text = Label(frame, text='format .png')
text.pack()

button1 = Button(frame, text='Choose files', command=chooseFiles)
button1.pack()

button2 = Button(root, text='Load files', command=lambda:[loadFiles(), predictFiles(), root.iconify()])
button2.pack(expand=YES)

root.mainloop()
