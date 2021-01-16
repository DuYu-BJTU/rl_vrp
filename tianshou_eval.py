import argparse

from envs.LVRP import LVRP
import tianshou as ts

import torch
from models.dqn import DQNet


def model_eval(config: dict):
    env = LVRP(config)

    state_shape = env.observation_space["distance"].len * 3
    action_shape = env.action_space.len
    net = DQNet(state_shape, action_shape, device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)
    state_dict = torch.load('dqn.pth')
    policy.load_state_dict(state_dict)
    policy.eval()
    policy.set_eps(0.05)
    collector = ts.data.Collector(policy, env)
    test = collector.collect(n_episode=1)
    print(test)


if __name__ == '__main__':
    if torch.cuda.is_available():
        # from utils.manager_torch import GPUManager
        #
        # gm = GPUManager()
        # device = torch.device("cuda:{}".format(gm.auto_choice()))
        device = torch.device("cuda:2")
    else:
        device = torch.device("cpu")

    small_config = {"loc_num": 3,
                    "cus_num": 4,
                    "cus_use_loc": 10,
                    "courier_inv": 30,
                    "loc_inv": 50,
                    "fp": 0.1,
                    "fd": 0.05,
                    "device": device}

    mid_config = {"loc_num": 5,
                  "cus_num": 6,
                  "cus_use_loc": 15,
                  "courier_inv": 40,
                  "loc_inv": 75,
                  "fp": 0.15,
                  "fd": 0.07,
                  "device": device}

    large_config = {"loc_num": 10,
                    "cus_num": 8,
                    "cus_use_loc": 20,
                    "courier_inv": 100,
                    "loc_inv": 200,
                    "fp": 0.2,
                    "fd": 0.1,
                    "device": device}

    parser = argparse.ArgumentParser(description='Dataset Choose')
    parser.add_argument("--dataset", default="small", type=str,
                        choices=["small", "mid", "large"], required=True,
                        dest="size")
    args = parser.parse_args()

    if args.size == "small":
        config = small_config
    elif args.size == "mid":
        config = mid_config
    elif args.size == "large":
        config = large_config
    else:
        raise KeyError("no such dataset {}".format(args.size))
    model_eval(config)
