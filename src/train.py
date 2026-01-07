import torch
import torch.nn as nn
import os
from datasets import walk_through, image_path, train_data, test_data
from torch.utils.data import DataLoader
from data import train_transform, test_transform, simple_transform, train_transform_trivial
from datasets import ImageFolderCustom, train_dir ,test_dir
from torchvision import datasets
from models.model3 import ImprovedCNN
from models.tinyVGG import TinyVGG
from config import class_names
#import torchinfo
#from torchinfo import summary
#from tqdm.auto import tqdm
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd



# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device
print(device)

# walking through the dataset 
walk_through(image_path)

## Turning loaded images into DataLoader's 
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=0,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             num_workers=0,
                             shuffle=False)

print(f"Length of train DataLoader: {len(train_dataloader)}")
print(f"Length of test DataLoader: {len(test_dataloader)}")

img, label = next(iter(train_dataloader))
print(f"Image Batch Shape: {img.shape}")
print(f"Label Batch Shape: {label.shape}")

train_data_custom = ImageFolderCustom(targ_dir =train_dir,
                                      transform=train_transform)

test_data_custom = ImageFolderCustom(targ_dir =test_dir,
                                     transform=test_transform)

# DataLoader's for custom datasets
train_dataloader_custom = DataLoader(
    dataset=train_data_custom,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True
)

test_dataloader_custom= DataLoader(
    dataset=test_data_custom,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False
)

img_custom, label_custom = next(iter(train_dataloader_custom))
print(f"Custom Image batch shape: {img_custom.shape}")
print(f"Custom Label batch shape: {label_custom.shape}")

train_data_simple = datasets.ImageFolder(
    root=train_dir,
    transform=simple_transform
)

test_data_simple = datasets.ImageFolder(
    root=test_dir,
    transform=simple_transform
)

NUM_COUNT = os.cpu_count()

train_dataloader_simple = DataLoader(
    dataset=train_data_simple,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True
)

test_dataloader_simple = DataLoader(
    dataset=test_data_simple,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False
)

torch.manual_seed(42)

model_0 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(class_names)).to(device)

# Create model_1 with data augmentation
train_data_augmented = datasets.ImageFolder(root=train_dir, transform=train_transform_trivial)
train_dataloader_augmented = DataLoader(
    dataset=train_data_augmented,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True
)

model_1 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(train_data_augmented.classes)).to(device)

print(model_0)

img_batch , label_batch = next(iter(train_dataloader_simple))

model_0(img_batch.to(device))
#summary(model_0, input_size=(BATCH_SIZE, 3, 224, 224))

# Testing function
def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
):
    model.train()

    train_loss, train_acc = 0, 0
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred) 

    train_loss = train_loss /len(dataloader)       
    train_acc = train_acc /len(dataloader)
    return train_loss, train_acc

#  TESting function 
def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device
):
    model.eval()

    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X,y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
        epochs: int =5,
        device=device
):
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(f"{epoch} | {train_loss:.4f} | {train_acc:.4f} | {test_loss:.4f} | {test_acc: .4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

# Execute training
torch.manual_seed(42)
torch.cuda.manual_seed(42)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)
start_time = timer()
model_0_results = train(
    model=model_0,
    train_dataloader=train_dataloader_simple,
    test_dataloader=test_dataloader_simple,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=5
    )
end_time = timer()
print(f"Total training time: {end_time - start_time:.3f} Seconds")

def plot_loss_curves(results: Dict[str, List[float]]):

    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="Training loss")
    plt.plot(epochs, test_loss, label="Testing Loss")

    plt.title("Plotting Of Loss Curves")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="Training Accuracy")
    plt.plot(epochs, test_accuracy, label="Testing Accuracy")
    plt.title("Plotting Of Accuracy Curves")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


plot_loss_curves(model_0_results)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.001)


start_time = timer()

model_1_results = train(
    model=model_1,
    train_dataloader=train_dataloader_augmented,
    test_dataloader=test_dataloader_simple,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=5
)

end_time = timer()

print(f"Total training time: {end_time - start_time:.3f} Seconds")
plot_loss_curves(model_1_results)

model_0_df = pd.DataFrame(model_0_results)
model_1_df = pd.DataFrame(model_1_results)

## Converting to charts 

epochs_model_0 = range(len(model_0_df))
epochs_model_1 = range(len(model_1_df))

plt.subplot(2, 2, 1)
plt.plot(epochs_model_0, model_0_df["train_loss"], color="red", label="model 0")
plt.plot(epochs_model_1, model_1_df["train_loss"], label="model 1")
plt.title("train_loss")
plt.xlabel("epochs")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs_model_0, model_0_df["test_loss"], color="magenta", label="model 0")
plt.plot(epochs_model_1, model_1_df["test_loss"], color="green", label="model 1")
plt.title("Test_loss")
plt.xlabel("epochs")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs_model_0, model_0_df["train_acc"], color="black", label="model 0")
plt.plot(epochs_model_1, model_1_df["train_acc"], color="orange", label="model 1")
plt.title("train_acc")
plt.xlabel("epochs")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs_model_0, model_0_df["test_acc"], color="purple", label="model 0")
plt.plot(epochs_model_1, model_1_df["test_acc"], color="blue", label="model 1")
plt.title("test_acc")
plt.xlabel("epochs")
plt.legend()

plt.tight_layout()
plt.show()

# Train ImprovedCNN (model_2)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

model_2 = ImprovedCNN(input_shape=3, hidden_units=64, output_shape=len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model_2.parameters(), lr=0.001)

start_time = timer()
model_2_results = train(
    model=model_2,
    train_dataloader=train_dataloader_simple,
    test_dataloader=test_dataloader_simple,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10
)
end_time = timer()
print(f"ImprovedCNN training time: {end_time - start_time:.3f} Seconds")

# Save the best model (ImprovedCNN)
model_save_path = "models"
os.makedirs(model_save_path, exist_ok=True)

# Save ImprovedCNN as the best model
torch.save(model_2.state_dict(), os.path.join(model_save_path, "best_model.pth"))
print(f"Best model saved to {model_save_path}/best_model.pth")

# Also save to backend directory
backend_model_path = os.path.join("..", "backend", "models")
os.makedirs(backend_model_path, exist_ok=True)
torch.save(model_2.state_dict(), os.path.join(backend_model_path, "best_model.pth"))
print(f"Model also saved to {backend_model_path}/best_model.pth")

plt.subplot(2, 2, 4)
plt.plot(epochs_model_0, model_0_df["test_acc"], label="model 0")
plt.plot(epochs_model_1, model_1_df["test_acc"], color="xkcd:sky blue", label="model 1")
plt.title("Test_acc")
plt.xlabel("epochs")
plt.legend()
plt.show()








### model 3
model_2 = ImprovedCNN(output_shape=len(class_names)).to(device)
optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=0.001, weight_decay=0.01)

start_time = timer()

model_2_results = train(
    model=model_2,
    train_dataloader=train_dataloader_simple,
    test_dataloader=test_dataloader_simple,
    optimizer=optimizer_2,
    loss_fn=loss_fn,
    epochs=10
)

end_time = timer()
print(f"Model 2 Training time : {end_time - start_time:.3f} Seconds")

plot_loss_curves(model_2_results)

model_2_df = pd.DataFrame(model_2_results)
epochs_model_2 = range(len(model_2_df))