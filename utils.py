from collections import Counter
import time

import numpy as np

def first_minimum(list):
    dict = Counter(list)
    return max(dict, key=dict.get)

def first_max_better(np_array):
    hist, _ = np.histogram(np_array)
    res = np.argmax(hist)
    return res

def first_max_better_int(np_array):
    return np.argmax(np.bincount(np_array))

def group_to_bins(list):
    pass