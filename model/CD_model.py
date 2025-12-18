# this file holds the main architecture for the RL model I hope to be able to train

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
        Input: 768 (embedding dimension)
        Hidden Layer 1: 256
        Hidden Layer 2: 128
        Hidden Layer 3: 64
        Output: TBD
        Activation: ReLU
        Dropout: 0.2 at each layer
"""

class CDModel(nn.Module):

    def __init__(self, input_dim=768, hidden_dim1=256, hidden_dim2=128, hidden_dim3=64, dropout_rate=0.2):
        super(CDModel, self).__init__()

        # Store hyperparameters
        
        self.input_dim = input_dim
        
        self.hidden_dim1 = hidden_dim2
        self.hidden_dim2 = hidden_dim3

        self.dropout_rate = dropout_rate

        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)

        # Define shared backbone layers
        self.fc1 = nn.Linear(input_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, hidden_dim3)

        # Two output heads (multi-head architecture)
       # self.output_layer_thread = nn.Linear(hidden_dim3, 1)  # Thread membership prediction
        self.output_layer_keep = nn.Linear(hidden_dim3, 1)    # Keep/discard prediction, might use later, likely not 

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # x shape: [batch_size, num_candidates, input_dim]

        # Self-attention layer
        attn_output, _ = self.attn(x, x, x)  # Query, Key, Value all from same input
        x = x + attn_output  # Residual connection

        # Shared backbone
        # Layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Two output heads
       # thread_logits = self.output_layer_thread(x).squeeze(-1)  # Shape: [B, N]
        keep_logits = self.output_layer_keep(x).squeeze(-1)      # Shape: [B, N]
        thread_logits = keep_logits # placeholder so this still runs 

        return thread_logits, keep_logits

    def get_model_config(self):
        """
        Returns a dictionary containing the model architecture configuration.
        Useful for logging and tracking experiments.

        Returns:
            dict: Model configuration parameters
        """
        config = {
            'architecture': 'CDModel (Multi-head + Self-Attention)',
            'input_dim': self.input_dim,
            'attention': {
                'num_heads': 4,
                'embed_dim': self.input_dim
            },
            'hidden_dim1': self.hidden_dim1,
            'hidden_dim2': self.hidden_dim2,
            'output_heads': {
                'thread_prediction': 1,
                'keep_discard_prediction': 1
            },
            'dropout_rate': self.dropout_rate,
            'activation': 'ReLU',
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        return config

"""
Example usage:

def example():
    model = CDModel()
    input_tensor = torch.rand(1, 10, 768)  # [batch_size=1, num_nodes=10, embedding_dim=768]
    thread_logits, keep_logits = model(input_tensor)
    print(f"Thread logits shape: {thread_logits.shape}")  # Should be [1, 10]
    print(f"Keep logits shape: {keep_logits.shape}")      # Should be [1, 10]

# example()
"""