

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ReplayBuffer:

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.gameplay_experiences = deque(maxlen=1000000)

    def __len__(self):
        return len(self.gameplay_experiences)

    def store_experience(self, state, next_state, reward, action, done):
        self.gameplay_experiences.append((state, next_state, reward, action, done))

    def sample_batch(self):
        sampled_gameplay_batch = random.sample(self.gameplay_experiences, self.batch_size)
        state_batch, next_state_batch, reward_batch, action_batch, done_batch = [], [], [], [], []
        for gameplay_experience in sampled_gameplay_batch:
            state_batch.append(gameplay_experience[0])
            next_state_batch.append(gameplay_experience[1])
            reward_batch.append(gameplay_experience[2])
            action_batch.append(gameplay_experience[3])
            done_batch.append(gameplay_experience[4])
        return state_batch, next_state_batch, reward_batch, action_batch, done_batch

    def sample_batch_with_uniform_action_selection(self, action_size):
        batch_size = self.batch_size
        samples_per_action = batch_size // action_size
        num_tries = 200
        sampled_gameplay_batch = []
        for action_idx in range(action_size):
            for _ in range(samples_per_action):
                for _ in range(num_tries):
                    sample = random.sample(self.gameplay_experiences, 1)[0]
                    if sample[3] == action_idx:
                        sampled_gameplay_batch.append(sample)
                        break
        residuals = batch_size - len(sampled_gameplay_batch)
        for _ in range(residuals):
            sample = random.sample(self.gameplay_experiences, 1)[0]
            sampled_gameplay_batch.append(sample)
        random.shuffle(sampled_gameplay_batch)
        assert len(sampled_gameplay_batch) == batch_size
        state_batch, next_state_batch, reward_batch, action_batch, done_batch = [], [], [], [], []
        for gameplay_experience in sampled_gameplay_batch:
            state_batch.append(gameplay_experience[0])
            next_state_batch.append(gameplay_experience[1])
            reward_batch.append(gameplay_experience[2])
            action_batch.append(gameplay_experience[3])
            done_batch.append(gameplay_experience[4])
        return state_batch, next_state_batch, reward_batch, action_batch, done_batch


def custom_normalization(unnormalized, min_ratio=.1):
    input_array = np.copy(unnormalized)
    input_array += np.ones_like(input_array) * (np.abs(np.min(input_array)))
    value = np.abs(np.max(input_array)) * min_ratio
    input_array += np.ones_like(input_array) * value
    normalizer = 1 / float(sum(input_array))
    output_array = np.array([x * normalizer for x in input_array])
    return output_array


class DQNTorch:

    def __init__(self, state_size, action_size, batch_size, input_size):

        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.input_size = input_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(self.input_size, self.action_size, self.device).float().to(self.device)
        self.target_net = DQN(self.input_size, self.action_size, self.device).float().to(self.device)
        self.update_target_network()
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)

    def seed(self, seed=42):
        torch.manual_seed(seed)

    def act_randomly(self):
        return torch.tensor([[random.randrange(self.action_size)]], device=self.device, dtype=torch.float)

    def create_single_action(self, action_index):
        return torch.tensor([[action_index]], device=self.device)

    def convert_action_index(self, value):
        value_tensor = torch.tensor([value], device=self.device, dtype=torch.long)
        return value_tensor

    def convert_reward(self, value):
        value_tensor = torch.tensor([value], device=self.device, dtype=torch.float)
        return value_tensor

    def act(self, state):
        state = state.view((-1,))
        with torch.no_grad():
            net_output = self.policy_net(state.float())
        return net_output.max(dim=0)[1]

    def act_roulette(self, state, min_ratio):
        state = state.view((-1,))
        with torch.no_grad():
            net_output = self.policy_net(state.float())
        net_output = net_output.cpu().detach().numpy()
        values = custom_normalization(net_output, min_ratio)
        action_index = np.random.choice(np.arange(values.shape[0]), p=values)
        return action_index

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=self.device, dtype=torch.long)

    def train(self, batch, GAMMA=.8):
        state_batch = torch.cat(batch[0])
        state_batch = state_batch.float().view(self.batch_size, -1)
        next_state_batch = batch[1]
        reward_batch = torch.cat(batch[2])
        action_batch = torch.cat(batch[3])

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in next_state_batch if s is not None])

        net_out = self.policy_net(state_batch)
        state_action_values = torch.zeros(self.batch_size, device=self.device)
        for i in range(self.batch_size):
            state_action_values[i] = net_out[i, action_batch[i]]

        non_final_next_states = non_final_next_states.float().view([-1, self.input_size])

        target_net_out = self.target_net(non_final_next_states)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        tmp_tensor = target_net_out.max(dim=1)[0].detach()
        next_state_values[:target_net_out.shape[0]] = tmp_tensor

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


def convert_to_network_input(state_array):
    network_input = torch.from_numpy(state_array)
    network_input = network_input.view((-1,))
    return network_input


class DQN(nn.Module):

    def __init__(self, input_size, action_size, device):
        super(DQN, self).__init__()

        self.device = device

        hidden_size = 64
        self.dense_1 = nn.Linear(input_size, hidden_size)
        self.dense_2 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        out = self.head(x)
        return out

