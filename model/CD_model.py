# this file holds the main architecture for the RL model I hope to be able to train

import torch
import torch.nn as nn
import torch.nn.functional as F


class CDModel(nn.Module):
    """
    Context Detection Model for conversation chunking.

    Architecture:
        Input: 768 (embedding dimension)
        Hidden Layer 1: 256
        Hidden Layer 2: 128
        Hidden Layer 3: 64
        Output: TBD
        Activation: ReLU
        Dropout: 0.2 at each layer
    """

    def __init__(self, input_dim=768, hidden_dim1=256, hidden_dim2=128, hidden_dim3=64, dropout_rate=0.2):
        super(CDModel, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 768)

        Returns:
            Output tensor of shape (batch_size, 64)
        """
        # Layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output layer will go here

        return x

