# Importing libraries 
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import requests
from pathlib import Path
import torch
import torch.nn as nn
import zipfile 
import os
import random
from PIL import Image
from data import dataTransform


# Importing the dataset 
data_path = Path("Data/")
image_path = data_path / "pizza-steak-sushi"

if image_path.is_dir() and any(image_path.iterdir()):
    print(f"{image_path} hai bhai")
else:
    image_path.mkdir(parents=True, exist_ok=True)
    
    with open(data_path/ "pizza-steak-sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip")
        print("kr rha")
        f.write(request.content)

    with zipfile.ZipFile(data_path/ "pizza-steak-sushi.zip", "r") as zip_ref:
        print("Nikal rha")
        zip_ref.extractall(image_path)

## Data Preparation
def walk_through(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"{len(dirnames)} directories and {len(filenames)} images hai")

## Loading image data using Imagefolder
train_dir = image_path / "train"
test_dir = image_path / "test"

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=dataTransform,
                                  target_transform=None)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=dataTransform)
print(train_data)
print(test_data)

# Create a custom dataset to replicate ImageFolder
class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir: str, transform=None):
        self.paths = list(Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = self.find_classes(targ_dir)
    
    def find_classes(self, directory: str):
        classes = sorted([entry.name for entry in list(os.scandir(directory)) if entry.is_dir()])
        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {directory}")
        class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
        return classes, class_to_idx

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx

            