import concurrent
import glob
import math
import os
import threading
from collections import Counter
from statistics import mean

import time
import numpy as np
import pyresample as pr
import satpy
from pandas import DataFrame
from satpy import Scene, DataQuery
from pprint import pprint
from pyresample import image, geometry
import matplotlib.pyplot as plt
import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3
import pandas
import pytz
import pyorbital
import tables

from utils import first_minimum, first_max_better, first_max_better_int


def load_seviri_data(file, reader, calibration="radiance", dataset="HRV"):
    scn = Scene(filenames={reader: [file]}, reader_kwargs={'calib_mode': 'GSICS'})
    scn.load([dataset], calibration=calibration)
    return scn


def print_seviri_data(file, reader, calibration, dataset="HRV"):
    scn = load_seviri_data(file, reader, calibration)
    scn.show(dataset)


def resample(scn, dataset, area_def, swath_def):
    values = scn[dataset].values
    values = values.astype("float32")

    resample_plot = pr.kd_tree.resample_nearest(swath_def,
                                                values,
                                                area_def,
                                                radius_of_influence=16000,  # in meters
                                                epsilon=0.5,
                                                fill_value=False)

    print("Resampled")
    plt.figure(figsize=(15, 15))
    plt.imshow(resample_plot)
    plt.show()


def sample_cloud_cover(scn, dataset, area_def, swath_def):
    values = scn[dataset].values
    values = values.astype("float32")
    zeros = np.zeros(values.shape)
    cloud_cover = np.where(values > 15, values, zeros)  # values[np.where(values > 20)]

    resampled = pr.kd_tree.resample_nearest(swath_def,
                                            cloud_cover,
                                            area_def,
                                            radius_of_influence=16000,  # in meters
                                            epsilon=0.5,
                                            fill_value=False)

    print("Plotted filter")
    plt.figure(figsize=(15, 15))
    plt.imshow(resampled)
    plt.show()


def sample_albedo(scn, dataset, area_def, swath_def):
    values = scn[dataset].values
    values = values.astype("float32")
    zeros = np.zeros(values.shape)
    filter = np.logical_and(values > 8, values < 12)
    albedo = np.where(filter, values, zeros)

    albedoed = pr.kd_tree.resample_nearest(swath_def,
                                           albedo,
                                           area_def,
                                           radius_of_influence=16000,  # in meters
                                           epsilon=0.5,
                                           fill_value=False)

    print("Albedo")
    plt.imshow(albedoed)
    plt.show()


def calculate_swath(scn, dataset):
    scn.load([dataset])
    lons, lats = scn[dataset].area.get_lonlats()
    lons = lons.astype("float32")
    lats = lats.astype("float32")
    return pr.geometry.SwathDefinition(lons=lons, lats=lats)


def create_europe_area():
    area_id = 'ease_sh'
    description = 'Antarctic EASE grid'
    proj_id = 'ease_sh'
    projection = {'proj': 'longlat', "ellps": "WGS84", "datum": "WGS84"}

    llx = -10  # lower left x coordinate in degrees
    lly = 35  # lower left y coordinate in degrees
    urx = 33  # upper right x coordinate in degrees
    ury = 75  # upper right y coordinate in degrees
    resolution = 0.005

    width = int((urx - llx) / resolution)
    height = int((ury - lly) / resolution)
    area_extent = (llx, lly, urx, ury)

    area_def = pr.geometry.AreaDefinition(area_id, description, proj_id, projection, width, height, area_extent)
    return area_def


def location_irradiance(scn, dataset, area, swath):
    values = scn[dataset].values
    values = values.astype("float32")
    lats_max = int(values.shape[0] / 20)
    lons_max = int(values.shape[1] / 20)
    irradiance_clear = np.zeros((values.shape[0], values.shape[1]))
    irradiance = np.zeros((values.shape[0], values.shape[1]))
    lons, lats = scn[dataset].area.get_lonlats()

    times = pandas.date_range(start='2020-04-09 09:00:00', end='2020-04-09 09:00:00', freq='15min')

    for y in range(lats_max):
        for x in range(lons_max):
            true_x = x * 20
            true_y = y * 20
            if lats[true_y][true_x] != math.inf and lons[true_y][true_x] != math.inf:
                result = Location(lats[true_y][true_x], lons[true_y][true_x]).get_clearsky(times).ghi.values
            else:
                result = 0
            for i in range(20):
                for j in range(20):
                    irradiance_clear[true_y + i][true_x + j] = result

    irradiance_clear_map = pr.kd_tree.resample_nearest(swath,
                                                       irradiance_clear,
                                                       area,
                                                       radius_of_influence=16000,  # in meters
                                                       epsilon=0.5,
                                                       fill_value=False)

    print("Irradiance")
    plt.figure(figsize=(15, 15))
    plt.imshow(irradiance_clear_map)
    plt.show()

    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            irradiance[y][x] = irradiance_clear[y][x] * (100 - values[y][x]) / 100

    irradiance_map = pr.kd_tree.resample_nearest(swath,
                                                 irradiance,
                                                 area,
                                                 radius_of_influence=16000,  # in meters
                                                 epsilon=0.5,
                                                 fill_value=False)

    print("Irradiance")
    plt.figure(figsize=(15, 15))
    plt.imshow(irradiance_map)
    plt.colorbar(label='$Irradiance W/m^2$')
    plt.clim(np.min(irradiance_map), np.max(irradiance_map))
    plt.show()


def calculate_irradiance_for(file, reader="seviri_l1b_native"):
    scene = Scene(filenames={reader: [file]})
    area = create_europe_area()
    swath = calculate_swath(scene, "HRV")
    resample(scene, "HRV", area, swath)
    sample_cloud_cover(scene, "HRV", area, swath)
    sample_albedo(scene, "HRV", area, swath)
    location_irradiance(scene, "HRV", area, swath)


def download_seviri_images(folder_path="./data/new/", dataset="HRV"):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, folder_path)
    data_files = os.listdir(data_path)
    data_files_filter = filter(lambda file: file.endswith(".nat"), data_files)
    return list(data_files_filter)


def threaded_albedo_calculation(
        prepared_data,
        dataset_size,
        x_start,
        x_range,
        cleansing_range = 25,
        folder_path="./data/new/",
        reader="seviri_l1b_native",
        dataset="HRV",
        calibration="radiance"):

    scene = load_seviri_data(folder_path + prepared_data[0], reader, "radiance")
    values = scene[dataset].values
    values = values.astype("float32")

    albedo_values = np.empty(shape=values.shape, dtype=object)
    for x in range(albedo_values.shape[0]):
        for y in range(albedo_values.shape[1]):
            albedo_values[x][y] = Counter()

    for i in range(dataset_size):
        print(i)
        scene = load_seviri_data(folder_path + prepared_data[i], reader, calibration)
        values = scene[dataset].values
        values = values.astype("float32")
        for x in range(x_start, x_range):
            for y in range(values.shape[1]):
                albedo_values[values.shape[0] - x - 1][values.shape[1] - y - 1].update({values[x][y] : 1})

    albedo_calc = np.empty(shape=values.shape, dtype="float32")
    for x in range(x_start, x_range):
        for y in range(values.shape[1]):
            albedo_calc[x][y] = first_max_better(albedo_values[values.shape[0] - x - 1][y])

    return albedo_calc


def albedo_calculation2(folder_path="./data/new/", reader="seviri_l1b_native", dataset="HRV", calibration="radiance"):
    start = time.time()
    prepared_data = download_seviri_images()
    scene = load_seviri_data(folder_path + prepared_data[0], reader, "radiance")
    values = scene[dataset].values
    values = values.astype("float32")


    result = np.empty(shape=values.shape, dtype=object)
    for x in range(values.shape[0]):
        for y in range(values.shape[1]):
            result[x][y] = []

    thread_range = int(values.shape[0]/4)
    first_start = 0
    first_end = thread_range
    second_end = 2 * thread_range
    third_end = 3 * thread_range
    end = values.shape[0]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        first_th = executor.submit(threaded_albedo_calculation,
            prepared_data=prepared_data,
            dataset_size=25,
            x_start=first_start,
            x_range=first_end)

        second_th = executor.submit(threaded_albedo_calculation,
            prepared_data=prepared_data,
            dataset_size=25,
            x_start=first_end,
            x_range=second_end)

        third_th = executor.submit(threaded_albedo_calculation,
            prepared_data=prepared_data,
            dataset_size=25,
            x_start=second_end,
            x_range=third_end)

        fourth_th = executor.submit(threaded_albedo_calculation,
            prepared_data=prepared_data,
            dataset_size=25,
            x_start=third_end,
            x_range=end)
    print("Test")
    first_thread = first_th.result()
    second_thread = second_th.result()
    third_thread = third_th.result()
    fourth_thread = fourth_th.result()
    for summator in range(first_start, first_end):
        for y in range(values.shape[1]):
            result[summator][y].append(first_thread[summator][y])

    for summator in range(first_end, second_end):
        for y in range(values.shape[1]):
            result[summator][y].append(second_thread[summator][y])

    for summator in range(second_end, third_end):
        for y in range(values.shape[1]):
            result[summator][y].append(third_thread[summator][y])

    for summator in range(third_end, values.shape[0]):
        for y in range(values.shape[1]):
            result[summator][y].append(fourth_thread[summator][y])


    albedo_values = np.empty(shape=values.shape, dtype="float32")
    for x in range(values.shape[0]):
        for y in range(values.shape[1]):
            albedo_values[albedo_values.shape[0] - x - 1][y] = first_minimum(result[x][y])

    end = time.time()
    print(end - start)
    plt.imshow(albedo_values)
    plt.show()




def albedo_calculation(folder_path="./data/new/", reader="seviri_l1b_native", dataset="HRV", calibration="radiance"):
    start = time.time()
    prepared_data = download_seviri_images()
    scene = load_seviri_data(folder_path + prepared_data[0], reader, "radiance")
    values = scene[dataset].values
    values = values.astype("float32")

    albedo_values = np.empty(shape=values.shape, dtype=object)
    albedo_calc = np.empty(shape=values.shape, dtype="float32")
    for x in range(albedo_values.shape[0]):
        for y in range(albedo_values.shape[1]):
            albedo_values[x][y] = Counter()


    for i in range(25):
        print(i)
        scene = load_seviri_data(folder_path + prepared_data[i], reader, calibration)
        values = scene[dataset].values
        values = values.astype("float32")
        for x in range(albedo_values.shape[0]):
            for y in range(albedo_values.shape[1]):
                albedo_values[albedo_values.shape[0] - x - 1][albedo_values.shape[1] - y - 1].update({values[x][y]: 1})

    for x in range(albedo_values.shape[0]):
        for y in range(albedo_values.shape[1]):
            albedo_calc[x][y] = first_max_better(albedo_values[x][y])

    print(albedo_calc.shape)
    end = time.time()
    print(end - start)
    plt.imshow(albedo_calc)
    plt.show()


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
    print("XD")
    start = time.time()
    result = np.apply_along_axis(first_max_better_int, 0, stacked)
    end = time.time()
    print(end - start)
    print("XD")
    plt.imshow(np.flip(result))
    plt.show()



if __name__ == '__main__':
    file1 = './data/new/MSG4-SEVI-MSG15-0100-NA-20210411085743.374000000Z-20210411085800-1499616.nat'
    file2 = './data/new/MSG4-SEVI-MSG15-0100-NA-20210415065743.958000000Z-20210415065800-1499616.nat'

    # calculate_irradiance_for(file1)
    # calculate_irradiance_for(file2)
    numpy_albedo()
    #albedo_calculation()
    #albedo_calculation2()
