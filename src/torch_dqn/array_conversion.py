

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


def zero_pad_to_dim(input_col_vec, target_dim=16):
    output_col_vec = np.zeros((target_dim, 1))
    output_col_vec[:len(input_col_vec), 0] = input_col_vec[:, 0]
    return output_col_vec

