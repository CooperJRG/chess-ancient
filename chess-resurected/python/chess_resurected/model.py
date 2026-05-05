"""AlphaZero-style network: residual tower, value head, policy head.

Dropout + L2 weight decay prevent the overfitting that kills training when
the replay buffer is small.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .features import NUM_INPUT_PLANES, NUM_POLICY


class _ResBlock(nn.Module):
    def __init__(self, channels: int, dropout_p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x), inplace=True)


class AlphaZeroNet(nn.Module):
    """Compact AlphaZero-style residual network.

    Args:
        num_blocks:  residual blocks (depth). Recommended: 4–8.
        channels:    feature-map width. Recommended: 64–256.
        dropout_p:   spatial dropout per residual block (0 = off).
    """

    def __init__(self, num_blocks: int = 6, channels: int = 128,
                 dropout_p: float = 0.1):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels   = channels
        self.dropout_p  = dropout_p

        self.input_proj = nn.Sequential(
            nn.Conv2d(NUM_INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(
            *[_ResBlock(channels, dropout_p) for _ in range(num_blocks)]
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(2 * 64, NUM_POLICY),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 18, 8, 8) float tensor
        Returns:
            value:  (B, 1) tanh-scaled win probability
            policy: (B, NUM_POLICY) raw logits — callers apply masked log_softmax
        """
        x = self.input_proj(x)
        x = self.tower(x)
        value  = self.value_head(x)
        policy = self.policy_head(x)
        return value, policy

    def predict(self, board_tensor: torch.Tensor, device: torch.device):
        """Single-position inference (eval mode, no grad)."""
        self.eval()
        with torch.no_grad():
            v, p = self(board_tensor.unsqueeze(0).to(device))
        return v.item(), p.squeeze(0)
