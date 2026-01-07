import flask_cors
import flask
from flask import flash, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import io
import sys 
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models.model3 import ImprovedCNN

app = flask.Flask(__name__)
CORS(app)
model = ImprovedCNN(
    input_shape=3,
    hidden_units=64,
    output_shape=3
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pth")

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    print(f"Model loaded from {MODEL_PATH}")

else:
    print(f"Model not found at {MODEL_PATH}")
model.eval()

transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

class_names = ["pizza", "steak", "sushi"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence = probabilities[0][predicted_class].item()
            predicted_class = torch.argmax(probabilities, dim=1).item()

        return jsonify({
            "prediction": class_names[predicted_class],
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                class_names[i]: 
                round(probabilities[0][i].item() * 100, 2)
                for i in range(len(class_names))}
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy"
    })

if __name__ == '__main__':
    app.run(debug=True,
            host='0.0.0.0',
            port=5000)
    
print("Sever is Ready")