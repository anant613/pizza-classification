import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import datasets
from data import train_transform_trivial
from datasets import train_dir, test_data_simple
from .tinyVGG import TinyVGG
from timeit import default_timer as timer


device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
# Create dataset first, then DataLoader
train_data_augmented = datasets.ImageFolder(root=train_dir, transform=train_transform_trivial)

train_dataLoader_augmentated = DataLoader(dataset=train_data_augmented,
                                          batch_size=32,
                                          num_workers=0,
                                          shuffle=True
                                          )
test_dataloader_augmentated = DataLoader(dataset=test_data_simple,
                                         batch_size=32,
                                         num_workers=0,
                                         shuffle=False
                                         )

model_1 = TinyVGG(
    input_shape=3,
    hidden_units=10,
    output_shape=len(train_data_augmented.classes)
).to(device)

print(model_1)