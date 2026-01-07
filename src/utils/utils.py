import torch 
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from typing import List
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    classNames: List[str],
    transform: torchvision.transforms = None,
    device: torch.device = device
):

 """Make a prediction on a target image with a trained model and plots the image and prediction"""

 target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
 target_image = target_image / 255

 if transform:
  target_image = transform(target_image)

 model.to(device)
 model.eval()
 with torch.inference_mode():
  target_image = target_image.unsqueeze(0)

  target_image_pred = model(target_image.to(device))

# Convert to logits
 target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

 target_image_pred_labels = torch.argmax(target_image_pred_probs, dim=1)

 plt.imshow(target_image.squeeze().permute(1, 2, 0))

 if classNames:
   title = f"Pred: {classNames[target_image_pred_labels]}"
 else:
   title = "Prediction"

 plt.title(title)
 plt.axis(False)

def predict_image(model, image, transform, class_names, device="cpu"):
    """Make prediction on a single PIL image for API use"""
    model.eval()
    with torch.no_grad():
        # Apply transform and add batch dimension
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Get all class probabilities
        all_probs = {
            class_names[i]: round(probabilities[0][i].item() * 100, 2)
            for i in range(len(class_names))
        }
        
        return {
            "prediction": class_names[predicted_class],
            "confidence": round(confidence * 100, 2),
            "probabilities": all_probs
        }
