import math
from collections import namedtuple
from itertools import count

import matplotlib
import matplotlib.pyplot as plt

import torch
from torch import optim
import torch.nn.functional as F

from tqdm import tqdm

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

    plot_durations(episode_durations)

    saved_info = {"config": config, "state_dict": policy_net.state_dict(),
                  "optimizer": optimizer.state_dict()}
    torch.save(saved_info, "attn.pth")
