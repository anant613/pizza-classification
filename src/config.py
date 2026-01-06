from datasets import train_data, test_data, train_dir, test_dir
import torch
import torch.nn as nn
import os

class_names = train_data.classes
class_dict = train_data.class_to_idx

print(class_names)
print(class_dict)

img, label = train_data[0][0], train_data[0][1]

target_directory = train_dir
class_name_found = sorted([entry.name for entry in list(os.scandir(target_directory))])

print(f"Target Directory: {target_directory}")
print(f"Classes found in directory: {class_name_found}")