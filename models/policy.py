import torch
from torch import nn


class AttnRouteChoose(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.input_dim = config["state_shape"]
        self.output_dim = config["action_shape"]
        self.h_dim = config["h_dim"]
        self.heads = config["heads"]
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
        self.device = config["device"]

    def forward(self, obs):

        obs = obs.transpose(1, 0)
        data_in = obs[:3]

        batch = data_in.shape[1]
        src_emb = data_in.shape[0]
        batch_in = data_in.reshape((-1, src_emb)).type(torch.float32).to(self.device)

        access = obs[-2]
        last_node = obs[-1].t()[0].unsqueeze(0).unsqueeze(-1).\
            expand(1, batch, self.h_dim).type(torch.int64).to(self.device)
        access = (1 - access).unsqueeze(1)
        access = access.repeat(self.heads, 1, 1).type(torch.bool).to(self.device)

        emb_out = self.encoder(batch_in).reshape((-1, batch, self.h_dim))
        query = torch.gather(emb_out, 0, last_node)
        attn_out, attn_weight = self.attn(query, emb_out, emb_out, attn_mask=access)
        # bn_out = self.bn((emb_out + attn_out).reshape((-1, self.h_dim)))
        # node_emb = self.bn(bn_out + self.feed(bn_out)).reshape((-1, batch, self.h_dim))
        logits = torch.squeeze(attn_weight)

        return logits
