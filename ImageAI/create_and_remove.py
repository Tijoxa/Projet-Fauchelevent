import os
import shutil

path = ""

def createValidationFolders():
    try :
        os.mkdir(path + "dataset/validation")
    except :
        pass
    
    L = os.listdir(path + "dataset/train")
    
    for label_name in L:
        try :
            os.mkdir(path + "dataset/validation/" + label_name)
        except :
            pass
        
        pourc = int(0.2*len(os.listdir(path + "dataset/train/" + label_name)))
        files = random.sample(os.listdir(path + "dataset/train" + label_name), pourc)
        
        for file in files :
            shutil.move(path + "dataset/train" + label_name + "/" + file, path + "dataset/validation/" + label_name) 

def removeValidationFolders():
    L = os.listdir(path + "dataset/validation")
    
    for label_name in L:
        for file in os.listdir(path + "dataset/validation" + label_name):
            shutil.move(path + "dataset/validation/" + label_name + "/" + file, path + "dataset/train" + label_name) 
    
    shutil.rmtree(path + "dataset/validation/")
