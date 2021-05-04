import os
from statistics import mean

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
    # cloud_cover_filter = values > 20
    zeros = np.zeros(values.shape)
    cloud_cover = np.where(values > 20, values, zeros)  # values[np.where(values > 20)]

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
    albedo = values[np.where(values > 2 and values < 20)]

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


def location_irradiance():
    pol = Location(50, 10, 'Etc/GMT-1', 34, 'Berlin')
    times = pandas.date_range(start='2020-04-09', end='2020-04-10', freq='15min')
    cs = pol.get_clearsky(times)
    cs.resample()
    plt.ylabel('Irradiance $W/m^2$')
    plt.title('Ineichen, climatological turbidity')
    plt.show()
    # Global Horizontal (GHI) = Direct Normal (DNI) X cos(θ) + Diffuse Horizontal (DHI)


if __name__ == '__main__':
    file2 = './data/new/MSG4-SEVI-MSG15-0100-NA-20210411085743.374000000Z-20210411085800-1499616.nat'
    reader = "seviri_l1b_native"
    scene = Scene(filenames={"seviri_l1b_native": [file2]})
    read_seviri_data(file2, reader, "radiance", "HRV")
    area = create_europe_area()
    resample(scene, "HRV", area, calculate_swath(scene, "HRV"))
    sample_cloud_cover(scene, "HRV", area, calculate_swath(scene, "HRV"))
    sample_albedo(scene, "HRV", area, calculate_swath(scene, "HRV"))
