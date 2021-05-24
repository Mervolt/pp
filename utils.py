from collections import Counter

def first_minimum(list):
    dict = Counter(list)
    return max(dict, key=dict.get)

def group_to_bins(list):
    pass