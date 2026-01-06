import torchvision
from torchvision import transforms
import os
from typing import Tuple, List, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Define a data transformation pipeline

dataTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# function to find class names in a target directory

def findClasses(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Find thr class folder names in a target directory"""
    classes = sorted([entry.name for entry in list(os.scandir(directory))])

    if not classes:
        raise FileNotFoundError(f"Ain't found {directory}")

    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

# findClasses(target_directory)

## Creating a transform for the dataset
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

simple_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

train_transform_trivial = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transform_trivial = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])