import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_size=4):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )

    def forward(self, x):
        return self.net(x)
