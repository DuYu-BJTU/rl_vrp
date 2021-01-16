import torch
import numpy as np
from torch import nn


class DQNet(nn.Module):
    def __init__(self, state_shape, action_shape, device):
        super().__init__()
        self.input_dim = state_shape
        self.output_dim = action_shape
        self.model = nn.Sequential(*[
            nn.Linear(self.input_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, self.output_dim), nn.Softmax(dim=1)
        ])
        self.device = device

    def forward(self, obs, state=None, info=None):
        if info is None:
            info = {}

        obs = obs.transpose((1, 0, 2))
        data_in = torch.from_numpy(obs[:3])
        data_in.to(self.device)

        batch = data_in.shape[1]
        batch_in = data_in.reshape(batch, -1).type(torch.float32)

        logits = self.model(batch_in)
        return logits, state
