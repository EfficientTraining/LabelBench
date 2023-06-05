import numpy as np


def accum_max(arr):
    return np.maximum.accumulate(arr)


def moving_avg(arr, window=3):
    arr = np.concatenate([np.ones(window - 1) * arr[0], arr])
    return np.convolve(arr, np.ones(window, dtype=float) / window, mode="valid")


def none_smooth(arr):
    return arr
