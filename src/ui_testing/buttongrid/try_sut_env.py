

from src.utils import path_utils
from src.ui_testing.buttongrid.sut_env import ButtonGrid


def try_manually():
    ButtonGrid()
    input('>Enter>')


if __name__ == '__main__':

    path_utils.alter_cwd_relative(level=3)

    try_manually()

