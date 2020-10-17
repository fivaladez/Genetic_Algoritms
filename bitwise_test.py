import numpy as np


if __name__ == '__main__':
    x = 134
    y = 203
    cut_index = 2
    rem_index = 8 - cut_index

    new_x = np.uint8(x >> rem_index) << rem_index
    new_x2 = np.uint8(y << cut_index) >> cut_index
    print(new_x, new_x2)
