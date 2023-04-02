

from src.utils import pickle_utils

from src.ui_testing.buttongrid.sut_env import BtgAction
from src.ui_testing.buttongrid.sut_env import compare_two_images
from src.controls import desktop_images


def states_are_equal(state1, state2):
    mse, ssim = compare_two_images(state1, state2)
    if abs(1 - ssim) < .005:
        return True
    else:
        return False


def custom_uniquify(list_obj):
    checked = []
    for e in list_obj:
        in_checked_marker = False
        for f in checked:
            if states_are_equal(e, f):
                in_checked_marker = True
                break
        if not in_checked_marker:
            checked.append(e)
    return checked


class StateActionExplorer:

    def __init__(self, storage_file_path=None, state_limit=None):
        if storage_file_path is None:
            self.storage_file_path = 'path\\to\\state_action_explorer_file_13_states.pkl'
        else:
            self.storage_file_path = storage_file_path

        self.state_limit = state_limit

        self.initial_state = None
        self.current_state = None
        self.states = []
        self.actions = []

    def set_current_state(self, state):
        self.current_state = state

    def set_initial_state(self, state):
        self.initial_state = state
        if len(self.states) == 0:
            self.states.insert(0, state)

    def clear(self):
        self.clear_states()
        self.clear_actions()

    def clear_states(self):
        self.states = []
        self.states.append(self.initial_state)

    def clear_actions(self):
        self.actions = []

    def save(self):
        store_actions = [x.to_string() for x in self.actions]
        storage_object = [self.initial_state, self.states, store_actions]
        pickle_utils.save(self.storage_file_path, storage_object)

    def load(self):
        load_storage = pickle_utils.load(self.storage_file_path)
        self.initial_state, self.states = load_storage[0], load_storage[1]
        self.actions = [BtgAction().from_string(x) for x in load_storage[2]]
        return self

    @staticmethod
    def actions_are_equal(action1, action2):
        return action1 == action2

    def register_action(self, action):
        actions_new = False
        already_in_actions = False
        for a in self.actions:
            if self.actions_are_equal(a, action):
                already_in_actions = True
                break
        if not already_in_actions:
            self.actions.append(action)
            actions_new = True
        return actions_new

    def register_state(self, state):
        states_new = False
        already_in_states = False
        for s in self.states:
            if states_are_equal(s, state):
                already_in_states = True
                break
        if not already_in_states:
            self.states.append(state)
            states_new = True
        return states_new

    def show_states(self):
        for s in self.states:
            desktop_images.show_grey_image(s)
        desktop_images.show()

    def show_actions(self):
        for a in self.actions:
            print(a.to_string())

    def set_actions(self, actions):
        self.actions = actions

    def get_all_actions(self):
        return self.actions

    def get_action(self, action_index):
        return self.actions[action_index]

    def get_action_index(self, action):
        return self.actions.index(action)

    def get_state(self, state_index):
        return self.states[state_index]

    def get_state_index(self, state):
        for i, s in enumerate(self.states):
            if states_are_equal(s, state):
                return i
        return None

    def uniquify_states(self):
        self.states = custom_uniquify(self.states)

