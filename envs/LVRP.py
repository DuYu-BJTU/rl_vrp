import argparse
from abc import ABC
from copy import deepcopy

import random
import torch
import gym
from gym import spaces
import numpy as np
from envs.dataset import data_generate


class MyList(spaces.Space, ABC):
    def __init__(self, values):
        assert len(values) >= 0
        if not isinstance(values, np.ndarray):
            self.values = np.array(values)
        else:
            self.values = values
        self.len = len(values)
        super(MyList, self).__init__(self.values.shape, self.values.dtype)

    def sample(self):
        return np.random.choice(self.values)

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)
        return x in self.values

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __getitem__(self, item):
        return self.values[item]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __repr__(self):
        return "MyList {}".format(self.values)

    def __eq__(self, other):
        return isinstance(other, MyList) and self.values == other.values


class LVRP(gym.Env, ABC):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config: dict):
        super(LVRP, self).__init__()
        self.locker_num = config["loc_num"]
        self.customer_per_region = config["cus_num"]
        self.customer_per_locker = config["cus_use_loc"]
        self.INV_MAX = config["courier_inv"]
        self.LOC_MAX = config["loc_inv"]
        self.fp = config["fp"]
        self.fd = config["fd"]
        self.device = config["device"]

        total_cus = self.customer_per_locker + self.customer_per_region

        while True:
            lockers, all_customer, split_cus, locker_idx = data_generate(loc=self.locker_num,
                                                                         cus=total_cus)
            counts = list()
            for idx in range(self.locker_num):
                counts.append(len(np.where(locker_idx == idx)[0]))
            if min(counts) > self.customer_per_region:
                break

        self.deliverys, self.pickups = self.delivery_pickup(locker_idx)
        self.max_locker_pick_time = self.pickups[0: self.locker_num] / self.fd

        cus_list = [split_cus[idx][:self.customer_per_region] for idx in range(len(split_cus))]  # 送货上门5个点 small sample
        self.cus_coord = np.concatenate([cus for cus in cus_list], axis=0)
        self.loc_coord = lockers
        self.all_coord = np.concatenate((self.loc_coord, self.cus_coord), axis=0)

        idx_range = self.locker_num + self.locker_num * self.customer_per_region
        self.access = np.ones(idx_range)
        self.last_access_rng = np.arange(idx_range)
        self.last_node = -1
        self.current = -1

        if self.device == torch.device('cpu'):
            from gym.envs.classic_control import rendering
            x, y = self.all_coord.transpose()
            x_len = max(x) - min(x)
            y_len = max(y) - min(y)
            self.viewer = rendering.Viewer(x_len, y_len)
            self.x_offset = -1 * min(x)
            self.y_offset = -1 * min(y)

        self.trace = list()

        # self.action_space = Integers(np.where(access == 1)[0])
        self.action_space = MyList(np.arange(idx_range))
        self.observation_space = self.at_dc(self.deliverys.copy(), self.pickups.copy(), np.zeros(idx_range))
        self.state = self.get_state()

        self.time = 0
        self.time_from_dc = 0

        self.inventory = 0
        self.total_pickup = 0

        self.loc_lost_sale = 0
        self.cus_lost_sale = np.zeros(idx_range)
        self.back_order = np.zeros(self.locker_num)
        self.back_order_cost = np.zeros(self.locker_num)
        self.backdc = 0
        self.work = 0

        self.loc_inv_d = np.zeros(self.locker_num)
        self.loc_inv_p = np.zeros(self.locker_num)
        self.loc_cache = np.zeros(self.locker_num)

    def render(self, mode='human', close=False):
        if self.device == torch.device('cpu'):
            from gym.envs.classic_control import rendering
            render_coord = self.all_coord + np.expand_dims(
                    [self.x_offset, self.y_offset],
                    axis=0).repeat(len(self.all_coord), axis=0)
            dc = rendering.make_circle(2)
            dc_trans = rendering.Transform(translation=render_coord[0])
            dc.add_attr(dc_trans)
            dc.set_color(255, 0, 0)
            self.viewer.add_geom(dc)

            for idx in range(1, 1 + self.locker_num):
                locker = rendering.make_circle(2)
                locker_trans = rendering.Transform(translation=render_coord[idx])
                locker.add_attr(locker_trans)
                locker.set_color(0, 255, 255)
                self.viewer.add_geom(locker)

            for idx in range(1 + self.locker_num, len(self.all_coord)):
                cus = rendering.make_circle(2)
                cus_trans = rendering.Transform(translation=render_coord[idx])
                cus.add_attr(cus_trans)
                cus.set_color(0, 0, 0)
                self.viewer.add_geom(cus)

        return self.viewer.render(return_rgb_array='rgb_array')

    def reset(self):
        self.time = 0
        self.work = 0
        self.last_node = -1
        self.loc_inv_d = np.zeros(self.locker_num)
        self.loc_inv_p = np.zeros(self.locker_num)
        time_list = np.zeros(len(self.all_coord))
        self.trace = list()

        total_cus = self.customer_per_locker + self.customer_per_region

        while True:
            lockers, all_customer, split_cus, locker_idx = data_generate(loc=self.locker_num,
                                                                         cus=total_cus)
            counts = list()
            for idx in range(self.locker_num):
                counts.append(len(np.where(locker_idx == idx)[0]))
            if min(counts) > self.customer_per_region:
                break

        self.deliverys, self.pickups = self.delivery_pickup(locker_idx)
        self.max_locker_pick_time = self.pickups[0: self.locker_num] / self.fd

        cus_list = [split_cus[idx][:self.customer_per_region] for idx in range(len(split_cus))]  # 送货上门5个点 small sample
        self.cus_coord = np.concatenate([cus for cus in cus_list], axis=0)
        self.loc_coord = lockers
        self.all_coord = np.concatenate((self.loc_coord, self.cus_coord), axis=0)

        self.loc_lost_sale = 0
        self.cus_lost_sale = np.zeros(len(self.all_coord))
        self.back_order = np.zeros(self.locker_num)
        self.back_order_cost = np.zeros(self.locker_num)
        self.backdc = 0
        self.observation_space = self.at_dc(self.deliverys.copy(), self.pickups.copy(), time_list)
        self.state = self.get_state()
        return self.state

    def step(self, action):
        idx = int(action)
        self.trace.append(idx)
        # For customer, loc_d & loc_p is accurate amount
        # For locker, loc_d & loc_p is the Total amount from individuals
        loc_d, loc_p = self.get_local_task(idx)

        # check if the courier inventory is enough for deliverys
        prev_inventory = self.inventory + loc_d
        if prev_inventory > self.INV_MAX:
            self.observation_space = self.back2dc(idx)
            self.trace.append(idx)
        distance = self.observation_space["distance"]
        time_list = self.observation_space["time"]
        self.time += distance[idx]
        self.time_from_dc += distance[idx]

        # calculate lockers pick up amount
        if idx in range(self.locker_num):
            # customers pick up their packages first
            self.loc_inv_d[idx] -= self.fp * (self.time - time_list[idx])
            if self.loc_inv_d[idx] < 0:
                self.loc_inv_d[idx] = 0

            # remaining capacity for customers to deliver their packages
            capacity = self.LOC_MAX - self.loc_inv_d[idx] - self.loc_inv_p[idx]

            # calculate how many packages customers can put into the locker
            if self.max_locker_pick_time[idx] >= self.time >= time_list[idx]:
                self.loc_cache[idx] = min(int(self.fd * (self.time - time_list[idx])), capacity)
            elif self.time >= self.max_locker_pick_time[idx] >= time_list[idx]:
                self.loc_cache[idx] = min(int(self.fd * (self.max_locker_pick_time[idx] - time_list[idx])), capacity)
            else:
                self.loc_cache[idx] = 0

            # packages failed to be delivered will become lost sale punishment
            self.loc_lost_sale += max(int(self.fd * (self.time - time_list[idx]) - self.loc_cache[idx]), 0)

            # put packages into lockers
            self.loc_inv_p[idx] += self.loc_cache[idx]

            # calculate how many packages can be taken by the courier
            rp_2 = self.INV_MAX - self.inventory + loc_d
            rp_1 = self.cus_lost_sale[idx] + self.loc_cache[idx]
            pickup = min(rp_1, rp_2, loc_p)

            self.cus_lost_sale[idx] = rp_1 - pickup

            loc_inv = self.loc_inv_d[idx] + self.loc_inv_p[idx]
            delivery = min(loc_d, self.LOC_MAX - loc_inv + pickup)
            self.back_order_cost[idx] += self.back_order[idx] * (self.time - time_list[idx])
            self.back_order[idx] = loc_d - delivery

            self.loc_inv_d[idx] += delivery
            self.loc_inv_p[idx] -= pickup

        else:
            rp_2 = self.INV_MAX - self.inventory + loc_d
            if loc_p > rp_2:
                self.cus_lost_sale[idx] = loc_p
                pickup = 0
            else:
                pickup = loc_p
            delivery = loc_d

        time_list[idx] = self.time
        deliverys = self.observation_space["delivery"]
        pickups = self.observation_space["pickup"]
        deliverys[idx] -= delivery
        pickups[idx] -= pickup
        self.work += delivery * self.time_from_dc + self.total_pickup * distance[idx]
        self.total_pickup += pickup

        last_access = deepcopy(self.access)

        self.access, access_rng = self.get_access(idx)
        loc = np.expand_dims(self.all_coord[idx], 0).repeat(len(self.all_coord), axis=0)
        distance = np.linalg.norm(self.all_coord - loc, axis=1)
        self.observation_space = spaces.Dict({
            "access": MyList(self.access),
            "distance": MyList(distance),
            "time": time_list,
            "delivery": deliverys,
            "pickup": pickups
        })
        if sum(deliverys) == 0 or 1 not in self.access:
            done = True
            self.observation_space = self.back2dc(idx, end=True)
        else:
            done = False

        reward = self.reward(action)
        self.last_access_rng = self.access.copy()
        self.last_node = idx
        self.state = self.get_state()
        cost = self.cost()
        return self.state, reward, done, {"access": self.access, "last_access": last_access,
                                          "delivery": deliverys, "trace": self.trace, "done": int(done),
                                          "cost": cost}

    def get_state(self):
        access = self.observation_space["access"].values.copy()
        distance = self.observation_space["distance"].values.copy()
        delivery = self.observation_space["delivery"].values.copy()
        pickup = self.observation_space["pickup"].values.copy()

        loc_idx = np.where(distance == 0)
        distance = np.expand_dims(distance, 0)
        delivery = np.expand_dims(delivery, 0)
        pickup = np.expand_dims(pickup, 0)

        if self.last_node != -1:
            last = self.last_node
        else:
            last = random.randint(self.locker_num, len(access) - 1)
        access = np.expand_dims(access, 0)
        last = np.array([last] * access.shape[1])
        last = np.expand_dims(last, 0)

        state = np.concatenate((distance, delivery, pickup, access, last), axis=0)
        return torch.from_numpy(state).unsqueeze(0)

    def delivery_pickup(self, locker_idx):
        d_c = np.random.randint(1, 7, len(locker_idx))
        p_c = np.random.randint(0, 4, len(locker_idx))

        lockers_d = list()
        lockers_p = list()
        customers_d = list()
        customers_p = list()

        for idx in range(self.locker_num):
            loc_idx = np.where(locker_idx == idx)[0]
            lockers_d.append(sum(d_c[loc_idx[self.customer_per_region:]]))
            lockers_p.append(sum(p_c[loc_idx[self.customer_per_region:]]))
            customers_d.append(d_c[loc_idx[:self.customer_per_region]])
            customers_p.append(p_c[loc_idx[:self.customer_per_region]])

        d = [np.array(lockers_d)] + customers_d
        p = [np.array(lockers_p)] + customers_p

        d = np.concatenate(d, axis=0)
        p = np.concatenate(p, axis=0)

        return d, p

    def back2dc(self, loc, end=False):
        self.backdc += 1
        deliverys = self.observation_space["delivery"].values
        pickups = self.observation_space["pickup"].values
        time_list = self.observation_space["time"].values
        observation_space = self.at_dc(deliverys, pickups, time_list)

        back_dist = self.observation_space["distance"][loc]
        self.time += back_dist
        if not end:
            self.time += back_dist

        return observation_space

    def get_local_task(self, action):
        deliverys = self.observation_space["delivery"]
        pickups = self.observation_space["pickup"]

        return deliverys[action], pickups[action]

    def get_access(self, action):
        if action in range(self.locker_num):
            access_rng = list(range(self.locker_num + self.customer_per_region * action,
                                    self.locker_num + self.customer_per_region * (action + 1)))
        else:
            region = int((action - self.locker_num) / self.customer_per_region)
            access_rng = list(range(self.locker_num + self.customer_per_region * region,
                                    self.locker_num + self.customer_per_region * (region + 1)))
            access_rng = list(range(self.locker_num)) + access_rng
            access_rng.remove(action)
            access_rng.remove(region)
        deliverys = self.observation_space["delivery"]
        total_tasks = deliverys.values
        total_inv = self.loc_inv_d + self.loc_inv_p
        access_rng = np.array(access_rng)
        access_idx = np.where(total_tasks[access_rng] != 0)[0]
        access_rng = access_rng[access_idx]
        if access_rng.size != 0 and self.locker_num > access_rng[0]:
            loc_rng = access_rng[np.where(access_rng < self.locker_num)[0]]
            remove_rng = np.where(total_inv[loc_rng] == self.INV_MAX)[0]
            access_rng = np.delete(access_rng, remove_rng)
        if access_rng.size == 0:
            access_rng = np.where(total_tasks[:self.locker_num] != 0)[0]
            rng_inv = total_inv[access_rng]
            access_rng = access_rng[np.where(rng_inv < self.INV_MAX)[0]]
        if access_rng.size != 0 and self.locker_num > access_rng[0]:
            loc_rng = access_rng[np.where(access_rng < self.locker_num)[0]]
            remove_rng = np.where(total_inv[loc_rng] == self.INV_MAX)[0]
            access_rng = np.delete(access_rng, remove_rng)

        if access_rng.size == 0:
            access_rng = np.where(total_tasks != 0)[0]
        access = np.zeros(len(self.all_coord))
        access[access_rng] = 1
        # self.action_space = Integers(access_rng)
        return access, access_rng

    def at_dc(self, deliverys, pickups, time_list):
        self.trace.append(-1)
        self.inventory = 0
        self.time_from_dc = 0
        self.total_pickup = 0
        self.access = np.concatenate([np.ones(self.locker_num),
                                      np.zeros(len(self.all_coord) - self.locker_num)],
                                     axis=0)
        distance = np.linalg.norm(self.all_coord, axis=1)
        observation_space = {
            "access": MyList(self.access),
            "distance": MyList(distance),
            "time": MyList(time_list),
            "delivery": MyList(deliverys),
            "pickup": MyList(pickups)
        }
        return observation_space

    def cost(self):
        value = self.work / 10 + self.backdc + self.time + self.loc_lost_sale + \
                sum(self.cus_lost_sale) + sum(self.back_order_cost)
        return value

    def split_cost(self):
        value = {
            "work": self.work,
            "time": self.time,
            "turns": self.backdc,
            "lost_sale": self.loc_lost_sale + sum(self.cus_lost_sale),
            "back_order": sum(self.back_order_cost)
        }
        return value

    def reward(self, action):
        if not self.last_access_rng[action]:
            pns = 1000
        else:
            pns = 0
        value = self.cost() + 1e-7
        return 100 / (value + pns)


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
                    "courier_inv": 60,
                    "loc_inv": 100,
                    "fp": 10,
                    "fd": 5,
                    "device": device}

    mid_config = {"loc_num": 5,
                  "cus_num": 6,
                  "cus_use_loc": 15,
                  "courier_inv": 80,
                  "loc_inv": 150,
                  "fp": 15,
                  "fd": 7,
                  "device": device}

    large_config = {"loc_num": 10,
                    "cus_num": 8,
                    "cus_use_loc": 20,
                    "courier_inv": 100,
                    "loc_inv": 200,
                    "fp": 20,
                    "fd": 10,
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

    env = LVRP(config)
    env.reset()
    env.step(env.action_space.sample())
    print(env.observation_space)
    env.step(env.action_space.sample())
    print(env.observation_space)
