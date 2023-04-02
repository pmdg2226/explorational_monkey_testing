

import os
import copy
import numpy as np
from collections import deque

from src.utils import time_utils
from src.utils import print_logger

from src.ui_testing.buttongrid.rl_env import RLEnvButtongrid

from src.torch_dqn.dqn_module import DQNTorch
from src.torch_dqn.dqn_module import ReplayBuffer
from src.torch_dqn.dqn_module import convert_to_network_input
from src.torch_dqn.array_conversion import convert_to_binary_array, zero_pad_to_dim


# file_name = os.path.basename(__file__).replace('.py', '.log')
# print_logger.initialize_logger(file_name)
# my_logger = print_logger.getLogger()


def create_state_representation_multiple_time_steps(states_memory):
    representation = ''
    for x in states_memory:
        representation += str(x) + ','
    representation = representation[:-1]
    return representation


def register(self, state):
    if state not in self.states_list:
        self.states_list.append(state)
        state_index = len(self.states_list)-1
    else:
        state_index = self.states_list.index(state)
    return state_index


def train(**kwargs):

    chosen_env = kwargs['chosen_env']
    num_episodes = kwargs['num_episodes']
    episode_max_steps = kwargs['episode_max_steps']
    seed_value = kwargs['seed']
    batch_size = kwargs['batch_size']
    epsilon = kwargs['epsilon']
    min_ratio = kwargs['min_ratio']
    target_update_rate = kwargs['target_update_rate']
    feature_mode = kwargs['feature_mode']
    policy_mode = kwargs['policy_mode']
    mem_steps = kwargs['mem_steps']
    log_file_path = kwargs['log_file_path']
    log_experiment_description = kwargs['log_experiment_description']

    if log_file_path is not None:
        assert not os.path.exists(log_file_path)
        print_logger.clear_loggers()
        print_logger.initialize_logger(log_file_path)
        my_logger = print_logger.getLogger()
        print('=== Start of Experiment ===')
        for k, v in kwargs.items():
            print('{} = {}'.format(k, v))

    time_utils.tic()

    if chosen_env == 'btg':
        env = RLEnvButtongrid(episode_max_steps=episode_max_steps, seed_value=seed_value)
    elif chosen_env == 'eox':
        pass
    else:
        raise Exception

    buffer = ReplayBuffer(batch_size)

    agent = DQNTorch(
        state_size=None,
        action_size=env.action_size,
        batch_size=batch_size,
        input_size=env.action_size
    )

    env.seed(seed_value)
    agent.seed(seed_value)

    state_mem_list = []

    walking_state_counts = []
    selected_actions = []
    episode_rewards = []
    episode_scores = []
    num_explored_states = []

    min_num_experiences = batch_size * 1
    stm_memory = []
    training_counter = 0
    update_counter = 0

    for episode in range(num_episodes):

        states_mem = deque(maxlen=mem_steps)
        for _ in range(mem_steps-1):
            states_mem.append(0)

        walking_state_count = 0
        episode_reward = 0
        state_index = env.reset()

        for step in range(env.episode_max_steps + 1):

            states_mem.append(state_index)
            state_rep = create_state_representation_multiple_time_steps(states_mem)
            if state_rep not in state_mem_list:
                state_mem_list.append(state_rep)
            state_index_network = state_mem_list.index(state_rep)
            state_arr = convert_to_network_input(zero_pad_to_dim(convert_to_binary_array(state_index_network), env.action_size))

            if np.random.random() < epsilon:
                action_index = agent.act_randomly().item()
                print('{} agent.act_randomly()'.format(action_index))
            else:
                if policy_mode == 'egreedy':
                    action_index = agent.act(state_arr).item()
                elif policy_mode == 'roulette':
                    action_index = agent.act_roulette(state_arr, min_ratio).item()
                else:
                    raise Exception
                print('{} agent.act()'.format(action_index))

            action_index = int(action_index)
            selected_actions.append(action_index)

            next_state_index, reward, done, info = env.step(action_index, step)

            next_states_mem = copy.deepcopy(states_mem)
            next_states_mem.append(next_state_index)
            next_state_rep = create_state_representation_multiple_time_steps(next_states_mem)
            if next_state_rep not in state_mem_list:
                state_mem_list.append(next_state_rep)
            next_state_index_network = state_mem_list.index(next_state_rep)
            next_state_arr = convert_to_network_input(zero_pad_to_dim(convert_to_binary_array(next_state_index_network), env.action_size))

            reward = agent.convert_reward(reward)
            action_index = agent.convert_action_index(action_index)

            if done:
                next_state_arr = None

            stm_memory.append((state_arr, next_state_arr, reward, action_index, done))

            episode_reward += reward

            if reward > 0:
                walking_state_count += 1

            state_index = next_state_index

            if len(buffer) > min_num_experiences:

                training_counter += 1
                if training_counter % 1 == 0:
                    training_counter = 0
                    batch = buffer.sample_batch()
                    agent.train(batch)

                update_counter += 1
                if update_counter % target_update_rate == 0:
                    print('episode {} - target network update'.format(episode))
                    update_counter = 0
                    agent.update_target_network()

            if done:

                for elem in stm_memory:
                    buffer.store_experience(*elem)
                stm_memory = []

                walking_state_counts.append(walking_state_count)
                episode_rewards.append(episode_reward.item())
                score = len(env.unique_states)
                episode_scores.append(score)
                num_explored_states.append(len(env.sae.states))

                print('episode {}, score {}, episode_reward {}, num_explored_states {}'.format(episode, score, episode_reward, num_explored_states[-1]))
                print('selected_actions {}'.format(selected_actions))
                print('walking_state_counts {}'.format(walking_state_counts))
                print('episode_rewards {}'.format(episode_rewards))
                print('episode_scores {}'.format(episode_scores))
                print('num_explored_states {}'.format(num_explored_states))

                break

    time_utils.toc()

