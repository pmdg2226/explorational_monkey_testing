

import os


def alter_cwd_relative(context_file=__file__, level=2, verbosity=True):
    abs_path = os.path.abspath(context_file)
    path = abs_path
    for _ in range(level):
        path = os.path.dirname(path)
    os.chdir(path)
    if verbosity is True:
        print(os.getcwd())

