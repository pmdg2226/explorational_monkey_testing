

import numpy as np


def convert_to_binary_array(number, digits=12):
    if number > 2**digits-1:
        raise Exception
    binary_string = f'{number:012b}'
    state_array = np.zeros((digits,))
    for i in range(digits):
        state_array[i] = int(binary_string[i])
    state_array = state_array.reshape((-1, 1))
    assert state_array.shape == (digits, 1)
    return state_array


class PotentialAssessor:

    def __init__(self, state_action_explorer_obj, action_size, state_size_limit=200):
        self.sae = state_action_explorer_obj
        self.action_size = action_size
        self.state_size_limit = state_size_limit
        self.action_count_limit = 5
        self.state_counts = np.zeros((self.state_size_limit,))
        self.state_dependant_action_counts = np.zeros((self.state_size_limit, self.action_size))

    def determine_action_potential_vector(self, state_index):
        action_potential_vec = np.zeros((self.action_size, 1))
        action_count_vec = self.state_dependant_action_counts[state_index, :]
        for i in range(self.action_size):
            if action_count_vec[i] < self.action_count_limit:
                action_potential_vec[i, 0] = 1
            else:
                action_potential_vec[i, 0] = 0
        return action_potential_vec

    def produce_state_rep(self, state_index):

        state_bin_vec = convert_to_binary_array(state_index)

        state_vec = np.zeros((self.action_size, 1))
        state_vec[:len(state_bin_vec), 0] = state_bin_vec[:, 0]

        action_potential_vec = self.determine_action_potential_vector(state_index)
        state_rep = np.hstack([state_vec, action_potential_vec])

        state_rep_no_potential = np.zeros((self.action_size, 2))
        state_rep_no_potential[:len(state_bin_vec), 0] = state_bin_vec[:, 0]

        state_rep_potential_only = np.zeros((self.action_size, 2))
        state_rep_potential_only[:len(action_potential_vec), 1] = action_potential_vec[:, 0]

        return state_rep, state_rep_no_potential, state_rep_potential_only

    def register_action_execution(self, state_index, action_index, next_state_index):
        # unsure to upcount state or next state in this case, depends
        self.state_counts[state_index] += 1
        # self.state_counts[next_state_index] += 1
        self.state_dependant_action_counts[state_index, action_index] += 1

    def produce_reward(self, state_index, action_index, next_state_index, reward_mode):

        if reward_mode == 'major_state_change':
            # potential improvement: add executed action_index to a virtual copy of the counts, neglected for simplicity
            before_act_pot = self.determine_action_potential_vector(state_index)
            after_act_pot = self.determine_action_potential_vector(next_state_index)
            # num_new_potential_actions = np.sum(after_act_pot) - np.sum(before_act_pot)
            num_new_potential_actions = 0
            for i in range(self.action_size):
                if after_act_pot[i] == 1 and before_act_pot[i] == 0:
                    num_new_potential_actions += 1
            num_state_count = self.state_counts[state_index]
            return max(1, num_new_potential_actions) / max(1, num_state_count)
        elif reward_mode == 'minor_state_change':
            return 1/self.state_dependant_action_counts[state_index, action_index]
        elif reward_mode == 'no_state_change':
            return -1
        else:
            raise Exception

