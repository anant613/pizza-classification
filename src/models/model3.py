import torch
import torch.nn as nn

class ImprovedCNN(nn.Module):
    def __init__(self, 
                 input_shape=3,
                 hidden_units=64,
                 output_shape=3):
            super().__init__()

            self.conv_block_1 = nn.Sequential(
                  nn.Conv2d(input_shape, hidden_units, 3, padding=1),
                  nn.BatchNorm2d(hidden_units),
                  nn.ReLU(),
                  nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
                  nn.MaxPool2d(2)
            )

            self.conv_block_2 = nn.Sequential(
                  nn.Conv2d(hidden_units, hidden_units*2, 3, padding=1),
                  nn.BatchNorm2d(hidden_units*2),
                  nn.ReLU(),
                  nn.MaxPool2d(2)
            )

            self.classifier = nn.Sequential(
                  nn.AdaptiveAvgPool2d((1, 1)),
                  nn.Flatten(),
                  nn.Dropout(0.5),
                  nn.Linear(hidden_units*2, output_shape)
            )

    def forward(self, x):
            x = self.conv_block_1(x)
            x = self.conv_block_2(x)
            x = self.classifier(x)
      
            return x