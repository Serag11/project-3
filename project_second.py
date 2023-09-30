import os
import pandas as pd
import torchvision.transforms as transforms
import torchvision
from torch.utils import data
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from skimage import io
import torch.optim as optim
from torch.utils.data.dataset import TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

#############  HYPERPARAMETERS  #############
test_size = 0.1
valid_size = 0.2
keep_prob = 0.2
nb_epoch = 10
samples_per_epoch = 20000
save_best_only = True
learning_rate = 1.0e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#############  DATA Getting  #############
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # img_path = self.img_labels.iloc[idx, 0]
        image = io.imread(img_path)
        label1 = torch.tensor(float(self.img_labels.iloc[idx , 3]))
        label2 = torch.tensor(float((self.img_labels.iloc[idx , 4])))
        label1 = label1.numpy()
        label2 = label2.numpy()
        label = np.column_stack([label1,label2])
        label = torch.from_numpy(label)
        label=label.flatten(0,-1)        
        if self.transform:
            image = self.transform(image)
        return image, label

dataset = CustomImageDataset(annotations_file= r"/home/sg/Downloads/simulator-linux/data/driving_log.csv",
                             img_dir=r"/home/sg/Downloads/simulator-linux/data/IMG",
                             transform= transforms.ToTensor() )

trainvalid_dataset, testing_dataset = train_test_split(
    dataset, test_size=test_size, random_state=0)
train_dataset, validation_dataset = train_test_split(
    trainvalid_dataset, test_size=valid_size, random_state=0)

train_data_loader = DataLoader(
    train_dataset, batch_size=30, shuffle=True)
validation_data_loader = DataLoader(
    validation_dataset, batch_size=30, shuffle=True)
testing_data_loader = DataLoader(
    testing_dataset, batch_size=30, shuffle=True)

#    Architicture
class MyModel(nn.Module):
    def __init__(self, input_shape, keep_prob):
        super(MyModel, self).__init__()
        self.input_shape = input_shape
        self.keep_prob = keep_prob
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 24, kernel_size=5, stride=2),      # 34*34
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),        # 16*16
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),      # 7*7
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3),             # 5*5
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3),       # 3*3
            nn.ELU()
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(keep_prob),
            nn.Flatten(),
            nn.Linear(self.get_flatten_size(), 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 2)
        )
    def forward(self, x):
        x = x / 127.5 - 1.0     # Normalize the input
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def get_flatten_size(self):
        with torch.no_grad():
            x = torch.zeros(1, *self.input_shape)
            x = self.conv_layers(x)
            return x.view(1, -1).size(1)

#################  TRAINING  #################
def train_model(model):
    """
    Train the model
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    model.train()
    best_loss = 100.0
    for epoch in range(nb_epoch):
        running_loss = 0.0

        for inputs, targets in train_data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_data_loader)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, targets in validation_data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
        valid_loss /= len(validation_data_loader)
        print(f'Epoch {epoch+1}/{nb_epoch} - Training Loss: {epoch_loss:.4f} - Validation Loss: {valid_loss:.4f}')

        # Save the best model
        if save_best_only and valid_loss < best_loss:
            best_loss = valid_loss
    print(f'Training completed with best loss = {best_loss:.4f}')

############  MAIN  ############
input_shape = (3, 160, 320)  # Assuming RGB images of size 224x224
keep_prob = 0.5  # Dropout keep probability
# model = MyModel(input_shape, keep_prob)
# train_model(model)


def imshow(imgs):
    imgs = imgs / 2 + 0.5   # unnormalize
    npimgs = imgs.numpy()
    plt.imshow(np.transpose(npimgs, (1, 2, 0)))
    plt.show()

# one batch of random training images
for i , (ig , l) in enumerate(testing_data_loader):
    if i == 1:
        img_grid = torchvision.utils.make_grid(ig[0:25], nrow=5)
        imshow(img_grid)
