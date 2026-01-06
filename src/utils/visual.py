import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import img, label, class_names, class_dict
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
from data import dataTransform
from explore_data import image_path_list, image_path
from requests.sessions import SessionRedirectMixin
from typing import List, Dict, Tuple
from train import train_data_custom, test_data_custom
from data import train_transform, test_transform

def plotTransformImages(image_paths, transform, n=3, seed=42):
    if seed is not None:
        random.seed(seed)

    sampled_paths = random.sample(image_paths, k=n)

    for img_path in sampled_paths:
        with Image.open(img_path) as img:
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))

            ax[0].imshow(img)
            ax[0].set_title(f"Original, image shape: {img.size}")
            ax[0].axis("off")

            transformed = transform(img)

            if hasattr(transformed, "permute"):
                # if normalized, undo normalization before plotting
                transformed = transformed.permute(1, 2, 0)

            ax[1].imshow(transformed)
            ax[1].set_title(f"Transformed , image shape: {transformed.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {img_path.parent.stem}")

    plt.show()

plotTransformImages(image_path_list, dataTransform, n=3, seed=None)


img_permut = img.permute(1, 2, 0)
print(f"Real image shape: {img_permut.shape}")
print(f"permuted image shape: {img_permut.shape}")

plt.figure(figsize=(10,7))
plt.imshow(img_permut)
plt.axis("off")
plt.title(f"Image permute {class_names[label]}")
plt.show()

def Display_random_images(dataset: torch.utils.data.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int= None):
    if n>10:
        n=10
        display_shape = False
        print("Can't be more than 10")

    if seed:
        random.seed(seed)
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    plt.figure(figsize=(15, 10))

    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        targ_image_adjust = targ_image.permute(1, 2, 0)
        plt.subplot(2, 5, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title += f" | shape: {targ_image_adjust.shape}"
            plt.title(title)
    plt.show()

Display_random_images(train_data_custom,
                      n=10,
                      classes=class_names,
                      seed=None)
plt.show()

plotTransformImages(image_path_list, train_transform, n=3)