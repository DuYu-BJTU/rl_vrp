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
            nn.Linear(self.input_dim[1], 128), nn.ReLU(inplace=True),
            nn.Linear(128, self.h_dim)
        ])
        self.attn = nn.MultiheadAttention(self.h_dim, self.heads)
        # self.feed = nn.Sequential(*[
        #     nn.Linear(self.h_dim, 512), nn.ReLU(inplace=True),
        #     nn.Linear(512, 512), nn.ReLU(inplace=True),
        #     nn.Linear(512, self.h_dim)
        # ])
        # self.bn = nn.BatchNorm1d(self.h_dim)
        # self.softmax = nn.Softmax(dim=1)
        self.device = device

    def forward(self, obs, state=None, info=None):
        if info is None:
            info = {}

        obs = obs.transpose((1, 0, 2))
        data_in = torch.from_numpy(obs[:3])
        data_in.to(self.device)

        batch = data_in.shape[1]
        src_emb = data_in.shape[0]
        batch_in = data_in.reshape((-1, src_emb)).type(torch.float32).to(self.device)

        access = obs[-2]
        last_node = obs[-1].transpose((1, 0))[0]
        last_node = torch.from_numpy(last_node).unsqueeze(0).unsqueeze(-1).\
            expand(1, batch, self.h_dim).type(torch.int64).to(self.device)
        access = torch.from_numpy(1 - access).unsqueeze(1)
        access = access.repeat(self.heads, 1, 1).type(torch.bool).to(self.device)

        emb_out = self.encoder(batch_in).reshape((-1, batch, self.h_dim))
        query = torch.gather(emb_out, 0, last_node)
        attn_out, attn_weight = self.attn(query, emb_out, emb_out, attn_mask=access)
        # bn_out = self.bn((emb_out + attn_out).reshape((-1, self.h_dim)))
        # node_emb = self.bn(bn_out + self.feed(bn_out)).reshape((-1, batch, self.h_dim))
        logits = torch.squeeze(attn_weight)

        return logits, state
