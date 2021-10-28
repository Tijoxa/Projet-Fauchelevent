## Import libraries
"""
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio===0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
"""

import numpy as np
import pandas as pd
import cv2
import os
import re
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from matplotlib import pyplot as plt

import xml.etree.ElementTree as ET

## Open and process files
path = 'C:/Users/DL/Documents/code/project with PyTorch/'
monRepertoire = "C:/Users/DL/Documents/code/dataset Global Wheat/train xml/"
readContentPath = 'C:/Users/DL/Desktop/00ab9e75ba68b9c4d37edd769c98fe728c61b5349915d30eb8f20298a40f5949.xml'
WEIGHTS_FILE = path+'fasterrcnn_resnet50_fpn_best.pth'

def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    list_with_labels = []

    for boxes in root.iter('object'):
        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        label = boxes.find("name").text

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

        list_with_labels.append(label)

    return filename, list_with_all_boxes, list_with_labels

name, boxes, labels = read_content(readContentPath)

from os import listdir
from os.path import isfile, join

fichiers = [join(monRepertoire, f) for f in listdir(monRepertoire) if isfile(join(monRepertoire, f))]
names = []  # liste de tous les noms de fichiers
boxes = []  # liste de liste de toutes les bboxes
labels = []

for file in fichiers :
    tmp1, tmp2, tmp3 = read_content(file)  # tmp2 les la liste des bboxes au format [[xmin, ymin, xmax, ymax], [[xmin, ymin, xmax, ymax], etc.]
    tmp1 = tmp1[:-4]  # nom des fichiers sans l'extension .png ou .jpg = image_id
    names += [tmp1]
    boxes += [tmp2]
    labels += [tmp3]

del tmp1, tmp2, tmp3  # plus besoin

train_df = pd.DataFrame({'image_id' : pd.Series(dtype='string'), 'xmin' : pd.Series(dtype='int'), 'ymin' : pd.Series(dtype='int'), 'xmax' : pd.Series(dtype='int'), 'ymax' : pd.Series(dtype='int'), 'labels' : pd.Series(dtype='string')})

for i in range(len(boxes)):
    for j in range(len(boxes[i])):  # marche po
        tmp = {'image_id': names[i], 'xmin': boxes[i][j][0], 'ymin': boxes[i][j][1], 'xmax': boxes[i][j][2], 'ymax': boxes[i][j][3], 'labels': labels[i][j]}
        train_df = train_df.append(tmp, ignore_index = True)

del tmp

train_df['xmin'] = train_df['xmin'].astype(float)
train_df['ymin'] = train_df['ymin'].astype(float)
train_df['xmax'] = train_df['xmax'].astype(float)
train_df['ymax'] = train_df['ymax'].astype(float)

image_ids = train_df['image_id'].unique()
trans = transforms.Compose([transforms.ToTensor()])

## Write a custom dataset (TO DO: support multiple classes, must be checked)
class CustomDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms = None, train = True):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.train = train
        self.labels = dataframe['labels']

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        image = Image.open(f'{self.image_dir}/{image_id}.jpg')
        image = np.array(image).astype(np.float32)
        image /= 255.0

        if self.transforms is not None:  # Apply transformation
            image = self.transforms(image)

        if (self.train == False):  # For test data
            return image, image_id

        # Else for train data
        records = self.df[self.df['image_id'] == image_id]
        boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        area = (boxes[:, 3]-boxes[:, 1])*(boxes[:, 2]-boxes[:, 0])
        area = torch.as_tensor(area, dtype = torch.float32)

        # Transforms 'Moisson' to 4 torch.int64 and so on
        tmp = []
        for i in range(records.shape[0]):
            if (labels[i] == 'Levee'):
                tmp += [1]
            elif (labels[i] == 'Tallage'):
                tmp += [2]
            elif (labels[i] == 'Epi'):
                tmp += [3]
            elif (labels[i] == 'Moisson'):
                tmp += [4]
        labels = torch.as_tensor(tmp, dtype = torch.int64)
        del tmp

        # Suppose all instances are not crowd, instances with iscrowd = True will be ignored during evaluation
        iscrowd = torch.zeros((records.shape[0],), dtype = torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['labels'] = labels

        return image, target, image_id

train_dir = path+'train'
test_dir = path+'testflv'

## For information
class Averager:  # Return the average loss
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

## Create model entries
def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = CustomDataset(train_df, train_dir, trans, True)

# Split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size = 16,
    shuffle = False,
    num_workers = 4,
    collate_fn = collate_fn
)

if torch.cuda.is_available() :
    device = torch.device('cuda')
    print('GPU available')
else :
    device = torch.device('cpu')
    print('GPU not available, run with CPU (slower)')

## Finetuning the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True, pretrained_backbone = False)

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(WEIGHTS_FILE, map_location = device))  # Load only weights and biaises

## Train the model
model.train()
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr = 0.03, momentum = 0.9, weight_decay = 0.00001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.5)

num_epochs = 20
epoch = 0

while (epoch < num_epochs) and (loss_hist.value > 0.2):
    loss_hist.reset()

    for i in range(len(train_data_loader.dataset)):
        image = [train_data_loader.dataset[i][0].to(device)]
        target = [{'boxes': train_data_loader.dataset[i][1]['boxes'].to(device), 'labels': train_data_loader.dataset[i][1]['labels'].to(device), 'iscrowd': train_data_loader.dataset[i][1]['iscrowd'].to(device), 'area': train_data_loader.dataset[i][1]['area'].to(device), 'image_id': train_data_loader.dataset[i][1]['image_id'].to(device)}]

        loss_dict = model(image, target)   # Return the loss
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        loss_hist.send(loss_value)  # Average out the loss
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # Update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Epoch #{epoch + 1} loss: {loss_hist.value}")

    epoch += 1

torch.save(model, path + "model_saved")

## Load the model
model = torch.load(path + "model_saved.pth", map_location = 'cpu')

## Prediction (TO DO: support prediction on GPU, test on images with differrent names)
detection_threshold = 0.5

trans = transforms.Compose([transforms.ToTensor()])
test_dataset = CustomDataset(submit, test_dir, trans, train = False)
test_data_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False)

def format_prediction_string(boxes, scores): ## Define the formate for storing prediction results
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)

outputs, results = [], []
model.eval()

for image, image_id in test_data_loader.dataset:  # Prediction is done image by image because of memory concerns
    outputs += model([image])

    for i, image in enumerate([image]):
        boxes = outputs[i]['boxes'].data.cpu().numpy()  # Formate of the output's box is [Xmin, Ymin, Xmax, Ymax]
        scores = outputs[i]['scores'].data.cpu().numpy()

        boxes = boxes[scores >= detection_threshold].astype(np.int32)  # Compare the score of output with the threshold and
        scores = scores[scores >= detection_threshold]  # Select only those boxes whose score is greater than threshold value
        image_id = image_id[i]

        boxes[:, 2] = boxes[:, 2]-boxes[:, 0]
        boxes[:, 3] = boxes[:, 3]-boxes[:, 1]  # Convert the box formate to [Xmin, Ymin, W, H]

        result = {  # Store the image id and boxes and scores in result dict.
            'image_id': image_id,
            'PredictionString': format_prediction_string(boxes, scores)
        }

        results.append(result)  # Append the result dict to Results list

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

## Display prediction of image IMG (TO DO: fix the issue of colors)
IMG = 9

sample = test_data_loader.dataset[IMG][0].permute(1,2,0).cpu().numpy()
for i in range(len(sample)):
    for j in range(len(sample[i])):
        sample[i][j][0] = 255*sample[i][j][0]
        sample[i][j][1] = 255*sample[i][j][1]
        sample[i][j][2] = 255*sample[i][j][2]

scores = outputs[IMG]['scores'].data.cpu().numpy()
boxes = outputs[IMG]['boxes'].data.cpu().numpy()
boxes = boxes[scores >= detection_threshold].astype(np.int32)

def rectangle(npimage, xmin, ymin, xmax, ymax, color = [0, 0, 0]):  # npimage in format (x, y, 3 colors)
    xmax = min(xmax, len(npimage)-1)
    ymax = min(ymax, len(npimage[0])-1)

    for i in range(xmin, xmax):
        npimage[i][ymin] = color
        npimage[i][ymax] = color

    for j in range(ymin, ymax):
        npimage[xmin][j] = color
        npimage[xmax][j] = color

    return npimage

for box in boxes :
    sample = rectangle(sample, box[0], box[1], box[2], box[3], [220, 0, 0])

sampleint = sample.astype(np.int32)

cv2.imwrite(path + 'output.png', sampleint)

## Submission
test_df.to_xml('submission.xml')  # Changed
