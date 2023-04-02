

import pynput


class DesktopActionExecuter:

    def __init__(self):
        self.mouse = pynput.mouse.Controller()
        self.keyboard = pynput.keyboard.Controller()

    def move(self, x, y):
        self.mouse.position = (x, y)

    def move_left_click(self, x, y):
        self.mouse.position = (x, y)
        self.mouse.press(pynput.mouse.Button.left)
        self.mouse.release(pynput.mouse.Button.left)

    def single_keystroke(self, key_string):
        self.keyboard.press(key_string)
        self.keyboard.release(key_string)

    def multiple_keystrokes(self, key_strings):
        for key_string in key_strings:
            self.single_keystroke(key_string)

