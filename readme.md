# Pizza–Steak–Sushi Image Classification (PyTorch)

## Overview

This project focuses on **multiclass image classification** using convolutional neural networks (**CNNs**) in **PyTorch**.
Three different architectures were implemented and compared to study the impact of **model capacity**, **data augmentation**, and **regularization** on generalization performance.

The task is to classify food images into **three classes**:

- **Pizza**
- **Steak**
- **Sushi**

## Dataset

- **Total images:** 300
- **Classes:** 3 (Pizza, Steak, Sushi)

### Train/Test split:

- **Train:** 225 images (75 per class)
- **Test:** 75 images (25 per class)

## Preprocessing

- Images resized to a **fixed resolution**
- **Normalization** applied
- **Data augmentation** (for selected models):
  - Random horizontal flip
  - Random rotation
  - **TrivialAugment**

## Project Structure

```
pizza-classification/
├── src/
│   ├── Data/
│   │   └── pizza-steak-sushi/
│   │       ├── train/
│   │       └── test/
│   ├── models/
│   │   ├── tinyVGG.py
│   │   ├── model2.py
│   │   └── model3.py
│   ├── utils/
│   │   ├── visual.py
│   │   └── models_comparison.py
│   ├── datasets.py
│   ├── data.py
│   ├── config.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Models Implemented

| Model | Key Characteristics | Purpose |
|-------|-------------------|----------|
| **TinyVGG** | Lightweight VGG-style CNN | Baseline performance |
| **TinyVGG + Augmentation** | Same architecture + data augmentation | Improve generalization |
| **ImprovedCNN** | Deeper CNN with BatchNorm, Dropout, AdaptiveAvgPool | Higher capacity and stability |

### Architectural Notes

- **AdaptiveAvgPool2d** was used to avoid hard-coded feature dimensions.
- **Batch Normalization** improved training stability.
- **Dropout** reduced overfitting in deeper models.

## Training Configuration

- **Framework:** PyTorch
- **Loss Function:** CrossEntropyLoss
- **Optimizers:**
  - **SGD** (TinyVGG)
  - **Adam** (TinyVGG + Aug)
  - **AdamW** (ImprovedCNN)
- **Epochs:** 10
- **Device:** CUDA (if available), otherwise CPU

## Results

### Model Comparison

| Model | Parameters | Test Accuracy |
|-------|------------|---------------|
| TinyVGG | ~450K | **78.3%** |
| TinyVGG + Augmentation | ~450K | **83.1%** |
| **ImprovedCNN** | ~2M | **🏆 87.4%** |

### Key Observations

- **Data augmentation** alone improved TinyVGG accuracy by **~5%**, highlighting its importance on small datasets.
- **ImprovedCNN** achieved the **highest test accuracy**, indicating better feature extraction and representation capacity.
- Although the deeper model required longer training time, it **generalized better** on unseen data.

## Best Model Selection

**ImprovedCNN** was selected as the final model because it achieved:

- ✅ The **highest test accuracy**
- ✅ More **stable training** due to BatchNorm
- ✅ **Reduced overfitting** due to Dropout
- ✅ **Architecture flexibility** via AdaptiveAvgPool2d

**Accuracy** was prioritized as the primary evaluation metric for this classification task.

## How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
cd src
python train.py
```

### Model Comparison
```bash
python utils/models_comparison.py
```

## Key Learnings

- **Data augmentation** significantly improves generalization on limited datasets.
- **Small CNNs** underfit complex visual patterns.
- **Regularization and normalization** are essential for deeper architectures.
- **Adaptive pooling** simplifies CNN design and prevents shape-related errors.

## Future Improvements

- **Transfer learning** with pre-trained models (ResNet, EfficientNet)
- **Hyperparameter tuning** (Optuna)
- **Model ensembling**
- **Deployment** using FastAPI + Docker

## Tech Stack

- **Python**
- **PyTorch**
- **torchvision**
- **matplotlib**
- **pandas**