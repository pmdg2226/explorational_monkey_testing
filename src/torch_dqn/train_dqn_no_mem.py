

import os
import numpy as np

from src.utils import time_utils
from src.utils import print_logger

from src.ui_testing.buttongrid.rl_env import RLEnvButtongrid

from src.torch_dqn.dqn_module import DQNTorch
from src.torch_dqn.dqn_module import ReplayBuffer
from src.torch_dqn.dqn_module import convert_to_network_input


# file_name = os.path.basename(__file__).replace('.py', '.log')
# print_logger.initialize_logger(file_name)
# my_logger = print_logger.getLogger()


def train(**kwargs):
    '''
    previous defaults: # def train(chosen_env=None, num_episodes=50, episode_max_steps=22, batch_size=32, epsilon=0, target_update_rate=50, feature_mode='with_potential', log_file_path=None, log_experiment_description=None):
    :param kwargs:
    :return:
    '''

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
        input_size=env.action_size*2
    )

    env.seed(seed_value)
    agent.seed(seed_value)

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

        walking_state_count = 0
        episode_reward = 0
        state_index = env.reset()

        for step in range(env.episode_max_steps + 1):

            state_arr = convert_to_network_input(env.get_state_rep(state_index, mode=feature_mode))

            if np.random.random() < epsilon:
                action_index = agent.act_randomly().item()
                print('{} agent.act_randomly()'.format(action_index))
            else:
                # action_index = agent.act(state_arr).item()
                action_index = agent.act_roulette(state_arr, min_ratio).item()
                print('{} agent.act()'.format(action_index))

            action_index = int(action_index)
            selected_actions.append(action_index)

            next_state_index, reward, done, info = env.step(action_index, step)

            next_state_arr = convert_to_network_input(env.get_state_rep(next_state_index, mode=feature_mode))

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

