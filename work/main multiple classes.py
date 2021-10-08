## Import libraries
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

## Open and process files
path = 'C:/Users/TM/Documents/Files/ML/fauchlevent/code/'

WEIGHTS_FILE = path+'fasterrcnn_resnet50_fpn_best.pth'
train_df = pd.read_csv(path+'train.csv')
submit = pd.read_csv(path+'sample_submission.csv')

train_df = train_df.drop(columns = ['width','height','source'])

train_df['image_id'].nunique()

train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)

image_ids = train_df['image_id'].unique()
trans = transforms.Compose([transforms.ToTensor()])

## Write a custom dataset (TO DO: support multiple classes, must be checked)
class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, labels, transforms = None, train = True):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.train = train
        self.labels = labels

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        image = Image.open(f'{self.image_dir}/{image_id}.jpg')
        image = np.array(image).astype(np.float32)
        image /= 255.0

        if self.transforms is not None:   # Apply transformation
            image = self.transforms(image)

        if (self.train == False):    # For test data
            return image, image_id

        # Else for train data
        records = self.df[self.df['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        area = (boxes[:, 3]-boxes[:, 1])*(boxes[:, 2]-boxes[:, 0])
        area = torch.as_tensor(area, dtype = torch.float32)

        # Support multiple classes, labels' shape is int64 (1, 2, 3, 4, 5) but we 'to_categorical" it
        tmp = np.zeros(labels + 1)
        tmp[labels] = 1
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

        return image, target, labels, image_id

train_dir = path+'train'
test_dir = path+'testflv'

## For information
class Averager:      ## Return the average loss
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

train_dataset = WheatDataset(train_df, train_dir, trans, True)

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
"""
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
"""

## Load the model
model = torch.load(path + "model_saved.pth", map_location = 'cpu')

## Prediction (TO DO: support prediction on GPU, test on images with differrent names)
detection_threshold = 0.5

trans = transforms.Compose([transforms.ToTensor()])
test_dataset = WheatDataset(submit, test_dir, trans, train = False)
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
test_df.to_csv('submission.csv', index = False)
