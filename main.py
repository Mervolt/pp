import math

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pyresample as pr
from pvlib.location import Location
from satpy import Scene
import time


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
    location_irradiance(scene, "HRV", area, swath)


if __name__ == '__main__':
    file1 = './data/new/MSG4-SEVI-MSG15-0100-NA-20210411085743.374000000Z-20210411085800-1499616.nat'
    file2 = './data/new/MSG4-SEVI-MSG15-0100-NA-20210415065743.958000000Z-20210415065800-1499616.nat'
    file3 = './data/ncef/MSG4-SEVI-MSG15-0100-NA-20200813235743.374000000Z-NA.nat'
    file4 = './data/ncef/MSG4-SEVI-MSG15-0100-NA-20200829172743.426000000Z-NA.nat'
    #print_seviri_data(file1, "seviri_l1b_native", "radiance")
    #print_seviri_data(file4, "seviri_l1b_native", "radiance")
    start = time.time()
    calculate_irradiance_for(file1)
    end = time.time()
    print(end - start)
    calculate_irradiance_for(file2)
