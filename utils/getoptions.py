import argparse

import torch


def get_options():
    if torch.cuda.is_available():
        # from utils.manager_torch import GPUManager
        #
        # gm = GPUManager()
        # device = torch.device("cuda:{}".format(gm.auto_choice()))
        device = torch.device("cuda:2")
    else:
        device = torch.device("cpu")

    small_config = {"loc_num": 3,
                    "cus_num": 5,
                    "cus_use_loc": 10,
                    "delivery_max": 5,
                    "pickup_max": 3,
                    "courier_inv": 80,
                    "loc_inv": 150,
                    "fp": 0.15,
                    "fd": 0.05,
                    "device": device}

    mid_config = {"loc_num": 5,
                  "cus_num": 10,
                  "cus_use_loc": 20,
                  "delivery_max": 5,
                  "pickup_max": 3,
                  "courier_inv": 150,
                  "loc_inv": 300,
                  "fp": 0.3,
                  "fd": 0.1,
                  "device": device}

    large_config = {"loc_num": 10,
                    "cus_num": 10,
                    "cus_use_loc": 20,
                    "delivery_max": 5,
                    "pickup_max": 3,
                    "courier_inv": 150,
                    "loc_inv": 300,
                    "fp": 0.3,
                    "fd": 0.1,
                    "device": device}

    parser = argparse.ArgumentParser(description='Dataset Choose')
    parser.add_argument("--dataset", default="small", type=str,
                        choices=["small", "mid", "large"], required=True,
                        dest="size")
    parser.add_argument("--batch", default=8, type=int, required=False, dest="batch")
    parser.add_argument("--h_dim", default=32, type=int, required=False, dest="h_dim")
    parser.add_argument("--heads", default=8, type=int, required=False, dest="heads")
    parser.add_argument("--update", default=10, type=int, required=False, dest="update_tgt")
    parser.add_argument("--epi", default=20000, type=int, required=False, dest="episode")
    parser.add_argument("--epoch", default=30, type=int, required=False, dest="epoch")
    parser.add_argument("--store", default=100, type=int, required=False, dest="store")
    args = parser.parse_args()

    if args.size == "small":
        config = small_config
    elif args.size == "mid":
        config = mid_config
    elif args.size == "large":
        config = large_config
    else:
        raise KeyError("no such dataset {}".format(args.size))

    config["batch"] = args.batch
    config["h_dim"] = args.h_dim
    config["heads"] = args.heads
    config["update_tgt"] = args.update_tgt
    config["epi_num"] = args.episode
    config["epoch"] = args.epoch
    config["store"] = args.store
    return config
