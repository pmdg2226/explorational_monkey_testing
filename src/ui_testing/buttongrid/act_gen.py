

import random


def choose_rand_coordinates(bbox):
    rand_x = random.randint(bbox[0], bbox[2])
    rand_y = random.randint(bbox[1], bbox[3])
    return rand_x, rand_y


def coordinates_within(bbox, x_coord, y_coord):
    c1 = (bbox[0] < x_coord < bbox[2])
    c2 = (bbox[1] < y_coord < bbox[3])
    if c1 and c2:
        return True
    else:
        return False


def choose_rand_string_input(mode='numbers'):
    length = random.randint(0, 4)
    if mode == 'numbers':
        chars_as_string = '1234567890'
    elif mode == 'chars':
        chars_as_string = 'abcdefghijklmnopqrstuvwxyz'
    else:
        raise Exception
    rand_input_string = ''
    for _ in range(length):
        char = random.choice(chars_as_string)
        rand_input_string += char
    return rand_input_string


class ButtonGridActionFactory:

    def __init__(self):
        # self.bbox_input_area = 8, 32, 306, 150  # Fahreheit GUI
        self.bbox_input_area = 20, 55, 490, 565  # ButtonGrid full
        # self.bbox_input_area = 320, 55, 490, 565  # ButtonGrid ID area

        # env.perform_action('mouse_move', [20, 55])
        # env.perform_action('mouse_move', [490, 55])
        # env.perform_action('mouse_move', [490, 565])
        # env.perform_action('mouse_move', [20, 565])

    def produce_random_mouse_move_left_click(self):
        action_type = 'move_left_click'
        args = list(choose_rand_coordinates(self.bbox_input_area))
        return action_type, args

    def produce_mouse_move_left_click(self, x, y):
        action_type = 'move_left_click'
        assert coordinates_within(self.bbox_input_area, x, y)
        args = [x, y]
        return action_type, args

    def produce_random_multiple_keystrokes(self, mode='numbers'):
        rand_input = choose_rand_string_input(mode='numbers')
        action_type = 'multiple_keystrokes'
        args = [rand_input]
        return action_type, args

