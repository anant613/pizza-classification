from datasets import image_path
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image_path_list = list(image_path.glob("*/*/*.jpg"))
randomImagePath = random.choice(image_path_list)

image_class = randomImagePath.parent.stem
print(image_class)

img = Image.open(randomImagePath)
print(f"Random image path: {randomImagePath}")
print(f"class: {image_class}")

img_as_array  = np.array(img)
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)

plt.title(f"Class: {image_class} | Image shape: {img_as_array.shape}")
plt.axis(False)
#plt.show()

