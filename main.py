import math
import os
from statistics import mean

import time
import numpy as np
import pyresample as pr
import satpy
from pandas import DataFrame
from satpy import Scene
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


def read_seviri_data(file, reader, calibration, dataset):
    scn = Scene(filenames={reader: [file]}, reader_kwargs={'calib_mode': 'GSICS'})
    scn.load(["HRV"], calibration=calibration)
    scn.show("HRV")


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
    plt.imshow(resampled)
    plt.show()


def sample_albedo(scn, dataset, area_def, swath_def):
    values = scn[dataset].values
    values = values.astype("float32")
    zeros = np.zeros(values.shape)
    filter = np.logical_and(values > 14, values < 16)
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
    plt.imshow(irradiance_map)
    plt.show()


if __name__ == '__main__':
    file2 = './data/new/MSG4-SEVI-MSG15-0100-NA-20210411085743.374000000Z-20210411085800-1499616.nat'
    reader = "seviri_l1b_native"
    scene = Scene(filenames={"seviri_l1b_native": [file2]})
    print(scene.end_time)
    read_seviri_data(file2, reader, "radiance", "HRV")
    area = create_europe_area()
    swath = calculate_swath(scene, "HRV")
    # resample(scene, "HRV", area, swath)
    # sample_cloud_cover(scene, "HRV", area, swath)
    # sample_albedo(scene, "HRV", area, swath)
    location_irradiance(scene, "HRV", area, swath)
