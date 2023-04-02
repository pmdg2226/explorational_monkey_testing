

import os
import time
import json
import subprocess
import random
import numpy as np

from src.utils import pickle_utils
from src.controls import desktop_images
from src.controls.sut_env_interface import SutEnvironment
from src.controls.desktop_controls import DesktopActionExecuter
from src.controls.desktop_images import take_greyscale_screenshot


def choose_rand_coordinates(bbox):
    '''
    bbox defined as xmin, ymin, xmax, ymax
    '''
    rand_x = random.randint(bbox[0], bbox[2])
    rand_y = random.randint(bbox[1], bbox[3])
    return rand_x, rand_y


def compare_two_images(image1, image2):
    # image1_grey = desktop_images.convert_rbga_to_grey_scale(image1)
    # image2_grey = desktop_images.convert_rbga_to_grey_scale(image2)
    mse = desktop_images.compare_two_greyscale_images_mse(image1, image2)
    ssim = desktop_images.compare_two_greyscale_images_ssim(image1, image2)
    return mse, ssim


def convert_to_grey(color_image):
    return desktop_images.convert_rbga_to_grey_scale(color_image)


def show_image(color_image, non_blocking=True):
    grey_image = convert_to_grey(color_image)
    desktop_images.show_rbga_grey_image(grey_image)
    if not non_blocking:
        desktop_images.show()


def show():
    desktop_images.show()


def choose_rand_coordinates(bbox):
    '''
    bbox defined as xmin, ymin, xmax, ymax
    '''
    rand_x = random.randint(bbox[0], bbox[2])
    rand_y = random.randint(bbox[1], bbox[3])
    return rand_x, rand_y


class BtgAction:

    def __init__(self, type=None, args=None):
        self.type = type
        self.args = args

    def __eq__(self, other):
        c1 = self.type == other.type
        c2 = self.args == other.args
        return c1 and c2

    def tuple(self):
        return self.type, self.args

    def to_string(self):
        json_data = {"type": self.type, "args": self.args}
        str_rep = json.dumps(json_data)
        return str_rep

    def from_string(self, input_string):
        json_data = json.loads(input_string)
        self.type = json_data['type']
        self.args = json_data['args']
        return self


class BtgActionFactoryPredefined:

    def __init__(self):
        '''
        bbox defined as xmin, ymin, xmax, ymax
        '''
        self.bbox_input_area = 20, 55, 490, 565

    def produce_all(self):
        all_actions = [
            # self.prod_home_button(),

            self.prod_back_button(),
            self.prod_first_button(),
            self.prod_second_button(),
            self.prod_third_button(),

            self.prod_dummy_click_0(),
            self.prod_dummy_click_1(),
            self.prod_dummy_click_2(),
            self.prod_dummy_click_3(),

            self.prod_dummy_click_4(),
            self.prod_dummy_click_5(),
            self.prod_dummy_click_6(),
            self.prod_dummy_click_7(),

            self.prod_dummy_click_8(),
            self.prod_dummy_click_9(),
            self.prod_dummy_click_10(),
            self.prod_dummy_click_11(),

            self.prod_dummy_click_12(),
            self.prod_dummy_click_13(),
            self.prod_dummy_click_14(),
            self.prod_dummy_click_15(),

        ]
        return all_actions

    def prod_home_button(self):
        return BtgAction('move_left_click', [110, 150])

    def prod_back_button(self):
        return BtgAction('move_left_click', [110, 300])

    def prod_first_button(self):
        return BtgAction('move_left_click', [280, 150])

    def prod_second_button(self):
        return BtgAction('move_left_click', [280, 300])

    def prod_third_button(self):
        return BtgAction('move_left_click', [280, 450])

    def prod_dummy_click_0(self):
        return BtgAction('move_left_click', [110, 450])

    def prod_dummy_click_1(self):
        return BtgAction('move_left_click', [120, 450])

    def prod_dummy_click_2(self):
        return BtgAction('move_left_click', [130, 450])

    def prod_dummy_click_3(self):
        return BtgAction('move_left_click', [110, 460])

    def prod_dummy_click_4(self):
        return BtgAction('move_left_click', [120, 460])

    def prod_dummy_click_5(self):
        return BtgAction('move_left_click', [130, 460])

    def prod_dummy_click_6(self):
        return BtgAction('move_left_click', [110, 470])

    def prod_dummy_click_7(self):
        return BtgAction('move_left_click', [120, 470])

    def prod_dummy_click_8(self):
        return BtgAction('move_left_click', [100, 460])

    def prod_dummy_click_9(self):
        return BtgAction('move_left_click', [100, 460])

    def prod_dummy_click_10(self):
        return BtgAction('move_left_click', [100, 470])

    def prod_dummy_click_11(self):
        return BtgAction('move_left_click', [100, 470])

    def prod_dummy_click_12(self):
        return BtgAction('move_left_click', [90, 460])

    def prod_dummy_click_13(self):
        return BtgAction('move_left_click', [90, 460])

    def prod_dummy_click_14(self):
        return BtgAction('move_left_click', [90, 470])

    def prod_dummy_click_15(self):
        return BtgAction('move_left_click', [90, 470])


class BtgActionFactoryRandom:

    def __init__(self):
        '''
        bbox defined as xmin, ymin, xmax, ymax
        '''
        self.bbox_input_area = 20, 55, 490, 565

    def produce_move_left_click(self):
        args = list(choose_rand_coordinates(self.bbox_input_area))
        return BtgAction('move_left_click', args)


class ButtonGrid:

    def __init__(self):

        self.initial_state_store_path = 'path/to/src/ui_testing/buttongrid/stored_initial_screen_image.pkl'
        self.stdout_log_path = 'path/to//src/ui_testing/buttongrid/test.log'
        self.stderr_log_path = 'path/to//src/ui_testing/buttongrid/test-error.log'

        self.restore_initial_state_image()

        self.process_ids = []

        if not os.path.exists(self.stdout_log_path):
            open(self.stdout_log_path, 'a').close()

        if not os.path.exists(self.stderr_log_path):
            open(self.stderr_log_path, 'a').close()

        self.bap = BtgActionFactoryPredefined()

        self.initialize_gui()
        self.initialize_controls()
        time.sleep(.5)  # time needed to display gui on screen in order to take screenshot afterwards

    def store_current_screen_image_as_initial_state_image(self):
        current_screen = self.get_screen_image()
        pickle_utils.save(self.initial_state_store_path, current_screen)

    def get_stored_init_state_image(self):
        stored_initial_screen_image = pickle_utils.load(self.initial_state_store_path)
        return stored_initial_screen_image

    def restore_initial_state_image(self):
        self.initial_state_image = self.get_stored_init_state_image()

    def __del__(self):
        self.kill_process()

    def seed(self, seed=42):
        pass

    def reset(self):
        action = self.bap.prod_home_button()
        self.perform_action(action.type, action.args)
        if self.current_screen_image_is_initial_state_image():
            pass
        else:
            raise Exception

    def current_screen_image_is_initial_state_image(self):
        A = self.get_screen_image()
        B = self.get_stored_init_state_image()
        mse = (np.square(A - B)).mean(axis=None)
        if mse < .5:
            return True
        else:
            return False

    def kill_process(self):
        if len(self.process_ids) >= 1:
            sel_process = self.process_ids[0]
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(sel_process.pid)])
            self.process_ids.remove(sel_process)
            time.sleep(.2)

    def initialize_gui(self):

        venv_path = r'venv/Scripts/python.exe'
        script_path = r'src/ui_testing/buttongrid/gui_base.py'

        assert os.path.exists(script_path)

        if os.path.exists(self.stdout_log_path):
            os.remove(self.stdout_log_path)
        if os.path.exists(self.stderr_log_path):
            os.remove(self.stderr_log_path)

        with open(self.stdout_log_path, "wb") as out, open(self.stderr_log_path, "wb") as err:
            pro = subprocess.Popen([venv_path, script_path], stdout=out, stderr=err)

        self.process_ids.append(pro)

    def initialize_controls(self):
        self.action_obj = DesktopActionExecuter()
        self.action_method_mapping = {
            'mouse_move': self.action_obj.move,
            'move_left_click': self.action_obj.move_left_click,
            'single_keystroke': self.action_obj.single_keystroke,
            'multiple_keystrokes': self.action_obj.multiple_keystrokes,
        }

    def perform_action(self, action_type, action_args):
        method = self.action_method_mapping[action_type]
        method(*action_args)

    def get_screen_image(self):
        # bbox_screenshot = 20, 55, 490, 565  # ButtonGrid full area
        bbox_screenshot = 345, 55, 490, 565  # ButtonGrid ID area
        image = take_greyscale_screenshot(bbox_screenshot)
        return image

    def get_log_data(self):
        with open(self.stdout_log_path, "r") as out, open(self.stderr_log_path, "r") as err:
            out_content = out.read()
            err_content = err.read()
        return out_content, err_content

    def deliver_feedback(self):
        image = self.get_screen_image()
        out_content, err_content = self.get_log_data()
        feedback = [image, out_content, err_content]
        return feedback

