from envs.LVRP import LVRP
import tianshou as ts

import argparse

import torch
from models.dqn import DQNet

from torch.utils.tensorboard import SummaryWriter


def model_train(config: dict):
    writer = SummaryWriter('log/dqn')
    env = LVRP(config)

    train_envs = ts.env.SubprocVectorEnv(
            [lambda: LVRP(config) for _ in range(16)]
    )
    test_envs = ts.env.SubprocVectorEnv(
            [lambda: LVRP(config) for _ in range(100)]
    )

    state_shape = (env.observation_space["distance"].len, 3)
    action_shape = env.action_space.len
    net = DQNet(state_shape, action_shape, 32, 8, config["device"])
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)

    train_collector = ts.data.Collector(policy, train_envs, ts.data.ReplayBuffer(size=20000))
    test_collector = ts.data.Collector(policy, test_envs)

    result = ts.trainer.offpolicy_trainer(
            policy, train_collector, test_collector,
            max_epoch=20, step_per_epoch=1000, collect_per_step=100,
            episode_per_test=100, batch_size=64,
            train_fn=lambda epoch, env_step: policy.set_eps(0.1),
            test_fn=lambda epoch, env_step: policy.set_eps(0.05),
            stop_fn=lambda mean_rewards: mean_rewards >= 10000000.0,
            writer=writer)

    print(f'Finished training! Use {result["duration"]}')

    torch.save(policy.state_dict(), 'dqn.pth')
    return policy


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

    model_train(config)
