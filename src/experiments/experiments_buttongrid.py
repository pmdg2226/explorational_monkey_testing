

import itertools
import numpy as np
import matplotlib.pyplot as plt

from src.utils import path_utils

from src.torch_dqn.train_dqn_memory import train as train_mem
from src.torch_dqn.train_dqn_no_mem import train as train_no_mem

from src.experiments.com_plots import parse_log_file, plot_show, plot_results_multiple, plot_multiple_mean_std

import scipy.stats as stats


def try_out():

    train_no_mem(log_experiment_description=None, chosen_env='eox', num_episodes=2,
                 episode_max_steps=10, seed=42, batch_size=32, epsilon=0, min_ratio=.05, target_update_rate=50,
                 feature_mode='without_potential', policy_mode='roulette', log_file_path=None)

    train_mem(log_experiment_description=None, chosen_env='eox', num_episodes=2,
              episode_max_steps=10, seed=42, batch_size=32, epsilon=0, min_ratio=.05, target_update_rate=50,
              feature_mode='potential_only', mem_steps=5, policy_mode='roulette', log_file_path=None)


def create_log_file_names(base_path, run_number=1):

    prefix = '2022-11-23' + '_run_' + str(run_number) + '_'
    labels = ['without_potential', 'with_potential', 'potential_only', 'random', 'memory-based']
    ending = '.log'

    log_file_names = [prefix + labels[i] + ending for i in range(len(labels))]
    log_file_paths = [base_path + x for x in log_file_names]

    return log_file_paths, labels


def exp_btg_6(base_path, runs, variants, eps, steps):

    for run_i in range(1, runs+1):

        log_file_paths, labels = create_log_file_names(base_path=base_path, run_number=run_i)

        seed_value = 41 + run_i

        train_no_mem(log_experiment_description=labels[0], chosen_env='btg', num_episodes=eps,
                     episode_max_steps=steps, seed=seed_value, batch_size=32, epsilon=0, min_ratio=.05,
                     target_update_rate=50,
                     feature_mode='without_potential', policy_mode='roulette', log_file_path=log_file_paths[0])

        train_no_mem(log_experiment_description=labels[1], chosen_env='btg', num_episodes=eps, episode_max_steps=steps, seed=seed_value,
                     batch_size=32, epsilon=0, min_ratio=.05, target_update_rate=50,
                     feature_mode='with_potential', policy_mode='roulette', log_file_path=log_file_paths[1])

        train_no_mem(log_experiment_description=labels[2], chosen_env='btg', num_episodes=eps, episode_max_steps=steps, seed=seed_value,
                     batch_size=32, epsilon=0, min_ratio=.05, target_update_rate=50,
                     feature_mode='potential_only', policy_mode='roulette', log_file_path=log_file_paths[2])

        train_no_mem(log_experiment_description=labels[3], chosen_env='btg', num_episodes=eps,
                     episode_max_steps=steps, seed=seed_value, batch_size=32, epsilon=1, min_ratio=.05, target_update_rate=50,
                     feature_mode='potential_only', policy_mode='roulette', log_file_path=log_file_paths[3])

        train_mem(log_experiment_description=labels[4], chosen_env='btg', num_episodes=eps,
                  episode_max_steps=steps, seed=seed_value, batch_size=32, epsilon=0, min_ratio=.05, target_update_rate=50,
                  feature_mode='potential_only', mem_steps=5, policy_mode='roulette', log_file_path=log_file_paths[4])


def ana_btg_6(base_path, runs, variants, eps, steps):

    paths_labels_per_run = []
    for run_i in range(1, runs + 1):
        log_file_paths, labels = create_log_file_names(base_path=base_path, run_number=run_i)
        paths_labels_per_run.append((log_file_paths, labels))

    parsed_data_per_run = []
    for log_file_paths, labels in paths_labels_per_run:
        parsed_contents = [list(parse_log_file(x)) for x in log_file_paths]
        parsed_data_per_run.append(parsed_contents)

    num_states = [[[y[0]] for y in x] for x in parsed_data_per_run]
    num_states = np.array(num_states)
    num_states = num_states[:, :, 0, :]
    num_states = np.delete(num_states, 0, axis=0)

    plotting(num_states)
    # statistics(num_states)


def plotting(num_states):

    mean_num_states = np.mean(num_states, axis=0)
    median_num_states = np.median(num_states, axis=0)
    std_num_states = np.std(num_states, axis=0)
    print(mean_num_states.shape, std_num_states.shape)

    new_labels = ['binary states', 'binary and potential', 'potential features', 'random', 'memory-based']

    x = np.arange(1, mean_num_states.shape[1]+1, dtype=np.int)
    plot_multiple_mean_std(median_num_states, std_num_states, new_labels, x, None, save_path_without_ending='exps/btg/6/btg_num_states_based_on_10_runs', xlabel='episodes', ylabel='number of discovered states')
    plot_show()


def statistics(num_states):

    print(num_states.shape)

    ps = previous_shape = num_states.shape
    new_num_states = np.zeros((ps[1], ps[0] * ps[2]))
    print(new_num_states.shape)

    collected = []
    for i in range(ps[0]):
        collected.append(num_states[i, :, :])
    new_num_states = np.hstack(collected)
    print(new_num_states.shape)

    new_labels = ['binary states', 'binary and potential', 'potential only', 'random', 'memory-based']
    print(new_labels)
    results = {i: 0 for i in range(5)}
    for i, j in itertools.permutations(range(5), 2):
        p_value = stats.wilcoxon(x=new_num_states[i], y=new_num_states[j], alternative='greater').pvalue
        print(i, j, [new_labels[i]], [new_labels[j]], 'greater', p_value < .05, p_value)
        if p_value < .01:
            results[i] += 1
        else:
            results[i] -= 0
    print('-' * 12)
    for i in range(5):
        print(new_labels[i], results[i])


if __name__ == '__main__':

    path_utils.alter_cwd_relative(level=3)

    base_path = 'exps/btg/6/'
    runs, variants, eps, steps = 10, 5, 50, 50
    # exp_btg_6(base_path, runs, variants, eps, steps)
    ana_btg_6(base_path, runs, variants, eps, steps)

