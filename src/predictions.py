import torch 
import torch.nn as nn
import torchvision 
import matplotlib.pyplot as plt
from torch import float32
from torchvision import transforms
from train import model_1, device, class_names

custom_image_path = "src/Data/check.jpg"

custom_image_uint8 = torchvision.io.read_image(str(custom_image_path))
print(custom_image_uint8.shape)
print(custom_image_uint8.dtype)

plt.imshow(custom_image_uint8.permute(1, 2, 0))
plt.axis(False)
plt.show()

custom_image = torchvision.io.read_image(str(custom_image_path)).type(float32)/ 255

custom_image_transform= transforms.Compose([
    transforms.Resize(size=(224, 224))
])

custom_image_transformed = custom_image_transform(custom_image)
print(custom_image_transformed.shape)
plt.imshow(custom_image_transformed.permute(1, 2, 0))
plt.axis(False)
plt.show()

model_1.eval()
with torch.inference_mode():
    custom_image_pred = model_1(custom_image_transformed.unsqueeze(0).to(device))

custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)  
custom_image_pred_labels = torch.argmax(custom_image_pred_probs, dim=1)

print(f"Predicted Class: {class_names[custom_image_pred_labels]}")