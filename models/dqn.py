import torch
import numpy as np
from torch import nn


class DQNet(nn.Module):
    def __init__(self, state_shape, action_shape, h_dim, heads, device):
        super().__init__()
        self.input_dim = state_shape
        self.output_dim = action_shape
        self.h_dim = h_dim
        self.heads = heads
        self.encoder = nn.Sequential(*[
            nn.Linear(self.input_dim[0], 128), nn.ReLU(inplace=True),
            nn.Linear(128, self.h_dim)
        ])
        self.attn = nn.MultiheadAttention(self.h_dim, self.heads)
        self.feed = nn.Sequential(*[
            nn.Linear(self.h_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, self.h_dim)
        ])
        self.bn = nn.BatchNorm1d(self.h_dim)
        self.softmax = nn.Softmax(dim=1)
        self.device = device

    def forward(self, obs, state=None, info=None):
        if info is None:
            info = {}

        obs = obs.transpose((1, 0, 2))
        data_in = torch.from_numpy(obs[:3])
        data_in.to(self.device)

        batch = data_in.shape[1]
        src_emb = data_in.shape[-1]
        batch_in = data_in.reshape((-1, src_emb)).type(torch.float32)

        access = obs[-1]
        last_node = np.where(obs[-1] == 2)[1]
        access[last_node] = 0
        access = torch.from_numpy(1 - access).unsqueeze(1).repeat(self.h_dim).type(torch.bool)

        emb_out = self.encoder(batch_in).reshape((-1, batch, self.h_dim))
        query = emb_out[last_node].unsqueeze(0)
        attn_out, attn_weight = self.attn(query, emb_out, emb_out, attn_mask=access)
        # bn_out = self.bn((emb_out + attn_out).reshape((-1, self.h_dim)))
        # node_emb = self.bn(bn_out + self.feed(bn_out)).reshape((-1, batch, self.h_dim))
        logits = self.softmax(torch.squeeze(attn_weight))

        return logits, state
