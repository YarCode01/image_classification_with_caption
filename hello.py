import os
import torch
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import time
import random 


torch.cuda.is_available()
x = torch.load("/Users/david/Downloads/Assignment2/image_classification_with_caption/images_data.pt")

plt.imshow(x[1].permute(1, 2, 0))
plt.show()
import re
import pandas as pd
from io import StringIO
FILENAME = 'train.csv'
with open(FILENAME) as file:
    lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    df_train = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
import re
import pandas as pd
from io import StringIO
FILENAME = 'test.csv'
with open(FILENAME) as file:
    lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    df_test = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
for i in range(df_train.shape[0]):
    df_train.Labels[i] = [int(j) for j in df_train.Labels[i].split()]
max_i = 0
for i in df_train.Labels:
    max_i = max(max_i, max(i))
min_i = 19
for i in df_train.Labels:
    min_i = min(min_i, min(i))
min_i
1
for i in range(1, max_i+1):
    df_train[f'{i}'] = 0
for i in range(df_train.shape[0]):
    for j in df_train.Labels[i]:
        df_train[f"{j}"][i] = 1
df_train
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, images, csv, train, test):
        self.csv = csv # df_train
        self.train = train # boolean
        self.test = test # boolean
        self.images = images

        self.all_image_names = self.csv[:]['ImageID']
        self.captions = self.csv[:]['Caption']

        self.all_labels = np.array(self.csv.drop(['ImageID', 'Labels', 'Caption'], axis=1))

        self.train_ratio = int(0.85 * len(self.csv))
        self.valid_ratio = len(self.csv) - self.train_ratio

        # set the training data images and labels
        if self.train == True:
            print(f"Number of training images: {self.train_ratio}")
            self.image_names = list(self.all_image_names[:self.train_ratio])
            self.labels = list(self.all_labels[:self.train_ratio])
            # define the training transforms
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45)
            ])
        # set the validation data images and labels
        elif self.train == False and self.test == False:
            print(f"Number of validation images: {self.valid_ratio}")
            self.image_names = list(self.all_image_names[-self.valid_ratio:])
            self.labels = list(self.all_labels[-self.valid_ratio:])
            # define the validation transforms
            self.transform = transforms.Compose([
            ])
        # set the test data images and labels, only last 10 images
        # this, we will use in a separate inference script
        elif self.test == True and self.train == False:
            self.image_names = list(self.all_image_names[:])
            self.labels = list(self.all_labels[:])
             # define the test transforms
            self.transform = transforms.Compose([
            ])
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image = self.images[index]
        image = self.transform(image)
        targets = self.labels[index]
        caption = self.captions[index]
        
        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32),
            # 'caption' : caption
        }
train_data = ImageDataset(
    x, df_train, train=True, test=False
)
# validation dataset
valid_data = ImageDataset(
    x, df_train, train=False, test=False
)
train_loader = DataLoader(
    train_data, 
    batch_size=64,
    shuffle=True
)
# validation data loader
valid_loader = DataLoader(
    valid_data, 
    batch_size=64,
    shuffle=False
)
# Distribution of Labels
df_train
label_df = df_train.explode("Labels")
label_counts = label_df['Labels'].value_counts()
sorted_label_counts = label_counts.sort_index()
#sorted_label_counts.plot.bar()
plt.bar(sorted_label_counts.index, sorted_label_counts.values)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title("Label Distribution")
plt.yticks(np.arange(0, 24000, step=1000))
plt.xticks(sorted_label_counts.index, sorted_label_counts.index)
plt.grid()
plt.show()

sorted_label_counts
label_list = []

for i in df_train['Labels']:
    label_list.extend(i.split())
label_list = sorted([int(i) for i in label_list])
label_list = [str(i) for i in label_list]

label_set = set(label_list)
label_set = sorted([int(i) for i in label_set])
label_set
# Networks

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5, num_filters = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
        nn.Conv2d(3, int(64*num_filters), kernel_size=11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(int(64*num_filters), int(192*num_filters), kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(int(192*num_filters), int(384*num_filters), kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(int(384*num_filters), int(256*num_filters), kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(int(256*num_filters), int(256*num_filters), kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        )


        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(int(256*num_filters) * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.sigm = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.sigm(x)
        return x

def train(criterion, model, loader, optimizer, device=None):
    model.train()
    N = len(loader)
    for i, (images, labels) in enumerate(loader): 
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval_loss_and_error(criterion, model, loader, device=None):
    model.eval()
    l, accuracy, ndata = 0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            l += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()
            ndata += len(data)
    
    return l/ndata, accuracy/ndata*100
from torchvision import models
# Use the torchvision's implementation of ResNeXt, but add FC layer for a different number of classes (27) and a Sigmoid instead of a default Softmax.
class Resnext50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained=True).to(device=device)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()
 
    def forward(self, x):
        return self.sigm(self.base_model(x))
 
# Initialize the model
model = Resnext50(19)
model = model.to(device)
# Switch model to the training mode
model.train()
x_batch = x[:128]
y = model(x_batch)
y.shape
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }
def calculate_metrics(preds, target, threshold=0.5):
    predicted_labels = []
    for raw_pred in preds:
        if sum(raw_pred>threshold) == 0:
            temp = np.zeros((18))
            temp[np.argmax(raw_pred[1])] = 1
            predicted_labels.append(temp)
        else:
            predicted_labels.append(np.array(raw_pred > threshold, dtype=float))

    preds = np.array(predicted_labels, dtype=float)
    for pred in preds:
        if sum(pred != 0) == 0:
            print('wrong')
    return  {
            'f1_score': f1_score(y_true=target, y_pred=preds, average='samples'),
            }
device = None
model = AlexNet(num_classes=19).to(device=device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
for it in range(5):
    for i, data in enumerate(train_loader):
        data, target = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        # outputs_copy = torch.clone(outputs)
        # indeces = outputs_copy >= 0.5
        # indeces1 = outputs_copy < 0.5
        # outputs_copy[indeces] = 1
        # outputs_copy[indeces1] = 0
        print(i)
        # print((outputs_copy == target.to(None)).sum(axis=1) == len(outputs_copy[0]))
        # print(((outputs_copy == target.to(None)).sum(axis=1) == len(outputs_copy[0])).sum()/len(outputs_copy)*100)
        loss = criterion(outputs, target)
        
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
