

import gym
import time
import copy
import numpy as np

from src.ui_testing.buttongrid.sut_env import compare_two_images
from src.ui_testing.buttongrid.state_explorer import StateActionExplorer
from src.ui_testing.buttongrid.potential_assessor import PotentialAssessor
from src.ui_testing.buttongrid.sut_env import ButtonGrid
from src.ui_testing.buttongrid.sut_env import BtgActionFactoryRandom
from src.ui_testing.buttongrid.sut_env import BtgActionFactoryPredefined


class RLEnvButtongrid(gym.Env):

    def __init__(self, episode_max_steps=32, seed_value=42, state_size=40, action_cooldown_time_in_secs=.2, init_cooldown_time_in_secs=1):

        self.seed_value = seed_value
        self.state_size = state_size
        self.action_cooldown_time_in_secs = action_cooldown_time_in_secs

        self.bar = BtgActionFactoryRandom()
        self.bap = BtgActionFactoryPredefined()

        self.sae = StateActionExplorer()
        # self.sae.load()

        self.episode_max_steps = episode_max_steps

        self.set_actions_to_start_actions()
        self.action_size = len(self.sae.actions)

        self.pot = PotentialAssessor(self.sae, self.action_size)

        self.btg = ButtonGrid()

        self.sae.set_initial_state(self.btg.initial_state_image)
        self.sae.set_current_state(self.btg.deliver_feedback()[0])

        time.sleep(init_cooldown_time_in_secs)

    def render(self):
        pass

    def seed(self, seed=42):
        self.btg.seed(seed)
        np.random.seed(seed)

    def get_state_rep(self, state_index, mode='with_potential'):

        state_rep, state_rep_no_potential, action_potential_vec = self.pot.produce_state_rep(state_index)

        if mode == 'with_potential':
            return state_rep
        elif mode == 'without_potential':
            return state_rep_no_potential
        elif mode == 'potential_only':
            return action_potential_vec
        else:
            raise Exception

    def reset(self):
        self.btg.reset()
        initial_state_index = 0
        self.unique_states = {initial_state_index}  # assuming init state is index 0
        state_index = self.sae.get_state_index(self.sae.initial_state)
        assert state_index == initial_state_index
        time.sleep(self.action_cooldown_time_in_secs)
        return state_index

    def get_state(self):
        return self.btg.deliver_feedback()[0]

    def get_random_action_index(self):
        actions = self.sae.get_all_actions()
        action = np.random.choice(actions)
        action_index = self.sae.get_action_index(action)
        return action_index

    def set_actions_to_start_actions(self):
        for act in self.bap.produce_all():
            self.sae.register_action(act)

    @staticmethod
    def determine_state_similarity(state1, state2):
        mse, ssim = compare_two_images(state1, state2)
        Xi_1 = .91  # needs to be determined individually
        Xi_2 = .96  # needs to be determined individually
        if ssim < Xi_1:
            return 'major_state_change'
        elif Xi_1 <= ssim < Xi_2:
            return 'minor_state_change'
        elif Xi_2 <= ssim:
            return 'no_state_change'
        else:
            raise Exception

    def step(self, action_index, step_number):

        old_state = copy.deepcopy(self.sae.current_state)
        old_state_index = self.sae.get_state_index(old_state)
        assert old_state_index is not None

        action = self.sae.get_action(action_index)
        self.btg.perform_action(action.type, action.args)
        self.sae.register_action(action)
        time.sleep(self.action_cooldown_time_in_secs)

        new_state = self.btg.deliver_feedback()[0]
        self.sae.set_current_state(new_state)
        self.sae.register_state(new_state)
        new_state_index = self.sae.get_state_index(new_state)
        assert new_state_index is not None

        if new_state_index not in self.unique_states:
            self.unique_states.add(new_state_index)

        reward_mode = self.determine_state_similarity(old_state, new_state)
        reward = self.pot.produce_reward(old_state_index, action_index, new_state_index, reward_mode)
        self.pot.register_action_execution(old_state_index, action_index, new_state_index)

        if step_number >= self.episode_max_steps:
            done = True
        else:
            done = False

        info = {}

        return new_state_index, reward, done, info

