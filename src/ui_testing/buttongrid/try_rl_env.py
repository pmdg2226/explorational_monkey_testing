

from src.utils import path_utils

from rl_env import RLEnvButtongrid


def try_depth_search_sequence():

    depth_search_action_sequence_level2 = [
        1, 0, 2, 0, 3
    ]

    depth_search_action_sequence_level3 = [
        1, 1, 0, 2, 0, 3, 0, 0,
        2, 1, 0, 2, 0, 3, 0, 0,
        3, 1, 0, 2, 0, 3
    ]

    depth_search_action_sequence_level4 = [
        1, 1,    1, 0, 2, 0, 3, 0, 0, 0,
        1, 2,    1, 0, 2, 0, 3, 0, 0, 0,
        1, 3,    1, 0, 2, 0, 3, 0, 0, 0,
        2, 1,    1, 0, 2, 0, 3, 0, 0, 0,
        2, 2,    1, 0, 2, 0, 3, 0, 0, 0,
        2, 3,    1, 0, 2, 0, 3, 0, 0, 0,
        3, 1,    1, 0, 2, 0, 3, 0, 0, 0,
        3, 2,    1, 0, 2, 0, 3, 0, 0, 0,
        3, 3,    1, 0, 2, 0, 3, 0, 0, 0,
    ]

    env = RLEnvButtongrid()
    env.reset()

    steps = -1
    for action_index in depth_search_action_sequence_level2:
    # for action_index in depth_search_action_sequence_level3:
    # for action_index in depth_search_action_sequence_level4:
        steps += 1
        # print(env.get_state_rep(env.sae.get_state_index(env.get_state())))
        new_state_index, reward, done, info = env.step(action_index, steps)
        # print(env.get_state_rep(new_state_index))

    env.sae.uniquify_states()
    print(len(env.sae.states), 'len(env.sae.states)')


if __name__ == '__main__':

    path_utils.alter_cwd_relative(level=3)

    try_depth_search_sequence()

