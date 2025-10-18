import torch
import torch.nn as nn
import numpy as np

class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) Attention Block.

    This module adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass of SEAttention.
        Args:
            x: Tensor of shape (batch_size, channels, H, W)
        Returns:
            Re-weighted tensor of same shape.
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MLP(torch.nn.Module):
    """
    Multi-layer Perceptron (MLP) model with integrated SE attention.
    Designed for age prediction from methylation features.
    """
    def __init__(self, in_features,num_hidden_1=32,num_hidden_2=32,num_hidden_3=32,num_hidden_4=32):
        super().__init__()
        self.atten = SEAttention(1362)
        self.my_network = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(in_features, num_hidden_1, bias=False),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(num_hidden_1),
            torch.nn.Dropout(0.1),

            # 2nd hidden layer
            torch.nn.Linear(num_hidden_1, num_hidden_2, bias=False),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(num_hidden_2),
            torch.nn.Dropout(0.1),

            # 3nd hidden layer
            torch.nn.Linear(num_hidden_2, num_hidden_3, bias=False),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(num_hidden_3),
            torch.nn.Dropout(0.1),

            # 4nd hidden layer
            torch.nn.Linear(num_hidden_3, num_hidden_4, bias=False),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(num_hidden_4),
            torch.nn.Dropout(0.1),

            # 5nd hidden layer
            torch.nn.Linear(num_hidden_4, 1, bias=False),

        )

    def forward(self, x):
        """
        Forward pass of MLP.
        - Reshapes input into 4D tensor for SEAttention
        - Applies SE attention
        - Flattens and passes through MLP layers
        """
        try:
            x = np.reshape(x,(x.shape[0], 1362,6,3),order='F')
            x = self.atten(x)
            x = x.detach().numpy()
            x = np.reshape(x,(x.shape[0],-1),order='F')
            x = torch.tensor(x)
            x = self.my_network(x)
            return x.squeeze()
        except Exception:
            raise ValueError("Input shape mismatch: expected (batch_size, 1362*6*3).")
        