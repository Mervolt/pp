import os
import time

import matplotlib.pyplot as plt
import numpy as np
from satpy import Scene

from utils import first_max_better_int


def numpy_albedo(folder_path="./data/new/", reader="seviri_l1b_native", dataset="HRV", calibration="radiance"):
    start = time.time()
    prepared_data = download_seviri_images()
    all_values = []
    for data in prepared_data:
        scene = load_seviri_data(folder_path + data, reader, calibration)
        values = scene[dataset].values
        values = values.astype("int")
        all_values.append(values)
    end = time.time()
    print(end - start)
    start = time.time()
    stacked = np.stack(all_values)
    stacked[stacked < 0] = 0
    end = time.time()
    print(end - start)
    start = time.time()
    result = np.apply_along_axis(first_max_better_int, 0, stacked)
    end = time.time()
    print(end - start)
    plt.imshow(np.flip(result))
    plt.show()


def download_seviri_images(folder_path="./data/new/", dataset="HRV"):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, folder_path)
    data_files = os.listdir(data_path)
    data_files_filter = filter(lambda file: file.endswith(".nat"), data_files)
    return list(data_files_filter)


def load_seviri_data(file, reader, calibration="radiance", dataset="HRV"):
    scn = Scene(filenames={reader: [file]}, reader_kwargs={'calib_mode': 'GSICS'})
    scn.load([dataset], calibration=calibration)
    return scn


def print_seviri_data(file, reader, calibration, dataset="HRV"):
    scn = load_seviri_data(file, reader, calibration)
    scn.show(dataset)


def first_max_better_int(np_array):
    return np.argmax(np.bincount(np_array))


if __name__ == '__main__':
    numpy_albedo()