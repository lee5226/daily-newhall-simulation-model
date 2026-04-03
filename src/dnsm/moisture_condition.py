import numpy as np


def moisture_condition(soil_profile, n_row=8, n_col=8):
    """
    Return the soil moisture condition code for a slot-based profile.

    Returns:
        1: dry
        2: dry/moist
        3: moist
    """
    if len(soil_profile) != (n_row * n_col):
        raise ValueError("Incorrect input length")

    m = np.arange(1, n_row * n_col + 1).reshape(n_row, n_col)

    idx_1, idx_2, idx_3 = m[1, 0], m[2, 0], m[3, 0]
    idx = [idx_1 - 1, idx_2 - 1, idx_3 - 1]

    values = np.array(soil_profile)[idx]

    if np.sum(values) == 0:
        return 1
    if np.all(values > 0):
        return 3
    if np.any(values > 0) and not np.all(values > 0):
        return 2

    raise ValueError("Unexpected condition encountered")