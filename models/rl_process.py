import math
from collections import namedtuple
from itertools import count

import matplotlib
import matplotlib.pyplot as plt
import os

import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from envs.LVRP import LVRP
from models.policy import AttnRouteChoose
import random
import gym

# # set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())
    plt.savefig("durations.png")


def select_action(state, steps_done, policy_net, device):
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = policy_net(state).argmax().view(1, 1)
    else:
        access = torch.nonzero(state.transpose(1, 0)[-2, 0])
        idx = random.randrange(access.shape[0])
        action = access[idx].view(1, 1).type(torch.int64).to(device)

    return action, steps_done


def optimize_model(memory: ReplayMemory, policy_net, target_net, optimizer, device, batch_size: int):
    GAMMA = 0.999
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).type(torch.float32)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    optimizer.step()


def rl_process(env: gym.Env, config: dict):
    config["state_shape"] = (env.observation_space["distance"].len, 3)
    config["action_shape"] = env.action_space.len
    policy_net = AttnRouteChoose(config).to(config["device"])
    target_net = AttnRouteChoose(config).to(config["device"])
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)

    steps_done = 0
    episode_durations = list()

    store_step = int(config["epi_num"] / config["store"])

    for i_episode in tqdm(range(config["epi_num"]), desc="Episode", total=config["epi_num"]):
        # Initialize the environment and state

        state = env.reset()
        for t in count():
            # Select and perform an action
            action, steps_done = select_action(state, steps_done, policy_net, config["device"])
            next_state, reward, done, info = env.step(action.item())
            reward = torch.tensor([reward], device=config["device"])
            if done:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(memory, policy_net, target_net, optimizer, config["device"], config["batch"])
            if done:
                episode_durations.append(t + 1)
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % config["update_tgt"] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (i_episode + 1) % store_step == 0:
            saved_info = {"config": config, "state_dict": policy_net.state_dict(),
                          "optimizer": optimizer.state_dict()}
            torch.save(saved_info, "saved/attn_{}.pth".format(i_episode + 1))

    plot_durations(episode_durations)


def eval_plt(env: LVRP, env_idx: int, turns: int):
    plt.figure(figsize=(10, 10), dpi=200)
    x_dc = 0
    y_dc = 0

    x_rng, y_rng = env.all_coord.transpose()
    x_rng = np.append(x_rng, [0])
    y_rng = np.append(y_rng, [0])
    x_rng = max(x_rng) - min(x_rng)
    y_rng = max(y_rng) - min(y_rng)
    factor = x_rng * y_rng / 90000
    head_width = 3 * factor
    head_length = 6 * factor
    line_width = 0.2 * factor

    node_color = ["red", "chocolate", "orange", "olive", "yellow", "palegreen",
                  "seagreen", "cadetblue", "navy", "darkviolet", "deeppink"]
    random.shuffle(node_color)

    plt.scatter(x_dc, y_dc, s=200, color='k',
                marker='*', label='DC')

    for idx, (x, y) in enumerate(env.loc_coord):
        plt.scatter(x, y, color=node_color[idx],
                    marker=',', label='LOC_{}'.format(idx))

        for cus in range(env.customer_per_region):
            cus_idx = idx * env.customer_per_region + cus
            cus_x, cus_y = env.cus_coord[cus_idx]
            plt.scatter(cus_x, cus_y, color=node_color[idx],
                        marker='o', label='CUS_{}'.format(idx))

    colors = ['k', 'r', 'y', 'g', 'c', 'b', 'm']
    color_idx = 0

    for head, end in zip(env.trace[:-1], env.trace[1:]):
        if head == -1:
            head_x = head_y = 0
        else:
            head_x, head_y = env.all_coord[head]
        if end == -1:
            end_x = end_y = 0
        else:
            end_x, end_y = env.all_coord[end]

        color = colors[color_idx]
        plt.arrow(head_x, head_y, end_x - head_x, end_y - head_y,
                  length_includes_head=True,  # 增加的长度包含箭头部分
                  head_width=head_width, head_length=head_length, color=color,
                  linestyle='-', linewidth=line_width)

        if end == -1:
            color_idx += 1

    if not os.path.exists("figure"):
        os.mkdir("figure")
    plt.savefig("figure/Figure_{}_Turns_{}.png".format(env_idx, turns))
    plt.close()


def rl_eval(epi_num: int):
    with torch.no_grad():
        model_list = os.listdir("saved/")
        epi_turns = list()
        for model_name in model_list:
            epi_turn = int(model_name.split("_")[-1].split(".")[0])
            epi_turns.append(epi_turn)

        epi_turns = sorted(epi_turns)
        model_name = "saved/attn_{}.pth".format(epi_turns[-1])
        saved_info = torch.load(model_name)
        config = saved_info["config"]
        config["courier_inv"] = 60
        config["loc_inv"] = 100
        env = LVRP(config)
        policy_net = AttnRouteChoose(config).to(config["device"])
        policy_net.load_state_dict(saved_info["state_dict"])

        logs = []

        for i_episode in tqdm(range(epi_num), desc="Test Epi", total=epi_num):
            state = env.reset()
            for t in count():
                action = policy_net(state).argmax().view(1, 1)
                next_state, reward, done, info = env.step(action.item())
                state = next_state
                if done:
                    log = {"cost": env.split_cost(), "trace": env.trace, "total": env.cost()}
                    logs.append(log)
                    eval_plt(env, i_episode, t)
                    break

        costs = list()
        works = list()
        times = list()
        lost_sales = list()
        back_orders = list()
        for log in logs:
            costs.append(log["total"])
            works.append(log["cost"]["work"])
            times.append(log["cost"]["time"])
            lost_sales.append(log["cost"]["lost_sale"])
            back_orders.append(log["cost"]["back_order"])

        if not os.path.exists("output"):
            os.mkdir("output")
        data = [list(range(epi_num)), works, times, lost_sales, back_orders]
        data = np.array(data)
        np.savetxt("output/2th.csv", data, delimiter=",")


def seq_plt(x, y, color, name="", log=True):
    plt.figure()
    plt.xlabel("Training Episodes")
    if name:
        plt.ylabel(name)
    else:
        plt.ylabel("Cost")
    if log:
        plt.semilogy(x, y, color=color, marker='o')
    else:
        plt.plot(x, y, color=color, marker='o')
    if not os.path.exists("diagram"):
        os.mkdir("diagram")
    plt.savefig("diagram/{}.png".format(name))
    plt.close()


def seq_eval(env_num: int):
    with torch.no_grad():
        model_list = os.listdir("saved/")
        epi_turns = list()
        for model_name in model_list:
            epi_turn = int(model_name.split("_")[-1].split(".")[0])
            epi_turns.append(epi_turn)

        epi_turns = sorted(epi_turns)
        model_name = "saved/attn_{}.pth".format(epi_turns[0])
        saved_info = torch.load(model_name)
        config = saved_info["config"]
        config["courier_inv"] = 60
        config["loc_inv"] = 100
        envs = [LVRP(config)] * env_num

        logs = []
        for epi_turn in tqdm(epi_turns, desc="Seq", total=len(epi_turns)):
            model_name = "saved/attn_{}.pth".format(epi_turn)
            saved_info = torch.load(model_name)
            config = saved_info["config"]
            policy_net = AttnRouteChoose(config).to(config["device"])
            policy_net.load_state_dict(saved_info["state_dict"])

            for idx, env in enumerate(envs):
                state = env.reset(new=False)
                for t in count():
                    action = policy_net(state).argmax().view(1, 1)
                    next_state, reward, done, info = env.step(action.item())
                    state = next_state
                    if done:
                        log = {"cost": env.split_cost(), "trace": env.trace, "total": env.cost()}
                        logs.append(log)
                        # eval_plt(env, epi_turn, t)
                        break

        colors = ["red", "chocolate", "orange", "olive", "yellow", "palegreen",
                  "seagreen", "cadetblue", "navy", "darkviolet", "deeppink"]
        random.shuffle(colors)

        costs = [0.0] * len(epi_turns)
        for idx, log in enumerate(logs):
            costs[int(idx / env_num)] += log["total"]
        for idx, cost in enumerate(costs):
            costs[idx] = cost / env_num
        seq_plt(epi_turns, costs, colors[5], "Total Cost", False)

        works = [0.0] * len(epi_turns)
        for idx, log in enumerate(logs):
            works[int(idx / env_num)] += log["cost"]["work"]
        for idx, work in enumerate(works):
            works[idx] = work / env_num
        seq_plt(epi_turns, works, colors[0], "Work Cost", False)

        time = [0.0] * len(epi_turns)
        for idx, log in enumerate(logs):
            time[int(idx / env_num)] += log["cost"]["time"]
        for idx, t in enumerate(time):
            time[idx] = t / env_num
        seq_plt(epi_turns, time, colors[1], "Time", False)

        lost_sales = [0.0] * len(epi_turns)
        for idx, log in enumerate(logs):
            lost_sales[int(idx / env_num)] += log["cost"]["lost_sale"]
        for idx, lost_sale in enumerate(lost_sales):
            lost_sales[idx] = lost_sale / env_num
        seq_plt(epi_turns, lost_sales, colors[2], "Lost Sale", False)

        back_orders = [0.0] * len(epi_turns)
        for idx, log in enumerate(logs):
            back_orders[int(idx / env_num)] += log["cost"]["back_order"]
        for idx, back_order in enumerate(back_orders):
            back_orders[idx] = back_order / env_num
        seq_plt(epi_turns, back_orders, colors[3], "Back Order", False)

        if not os.path.exists("output"):
            os.mkdir("output")
        data = [epi_turns, works, time, lost_sales, back_orders]
        data = np.array(data)
        np.savetxt("output/4th.csv", data, delimiter=",")
