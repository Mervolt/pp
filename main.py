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


def read_seviri_data(scn, file, reader, calibration, dataset):
    scn = Scene(filenames={reader: [file]}, reader_kwargs={'calib_mode': 'GSICS'})
    scn.load([dataset], calibration=calibration)
    scn.show(dataset)
    print("XD")
    print(scn[dataset].area.get_lonlats())
    lon, lats = scn[dataset].area.get_lonlats()
    print("LON")
    print(len(lon))
    print(lon)
    print("LATS")
    print(len(lats))
    print(lats)
    print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
    #print(scn[dataset].values())
    print("SCN Values")
    pprint(vars(scn))


def resample(scn, dataset, area_def):
    print(scn.all_dataset_names())
    scn.load([dataset])
    lons, lats = scn[dataset].area.get_lonlats()
    lons = lons.astype("float32")
    lats = lats.astype("float32")
    swath_def = pr.geometry.SwathDefinition(lons=lons, lats=lats)
    values = scn[dataset].values
    values = values.astype("float32")
    resampled = pr.kd_tree.resample_nearest(swath_def, values,
                                            area_def,
                                            radius_of_influence=16000,  # in meters
                                            epsilon=0.5,
                                            fill_value=False)

    print("Resampled")
    plt.imshow(resampled)
    plt.show()
    new_scn = scn.resample("eurol")
    new_scn.load([dataset])
    new_scn.show(dataset)


def create_spain_area():
    area_id = "Spain"
    description = "Geographical Coordinate System clipped on Spain"
    proj_id = "Spain"

    proj_dict = {"proj": "longlat", "ellps": "WGS84", "datum": "WGS84"}

    llx = 3.3  # lower left x coordinate in degrees
    lly = 35.9  # lower left y coordinate in degrees
    urx = 13.3  # upper right x coordinate in degrees
    ury = 43.8  # upper right y coordinate in degrees
    resolution = 0.005

    width = int((urx - llx) / resolution)
    height = int((ury - lly) / resolution)
    area_extent = (llx, lly, urx, ury)

    area_def = pr.geometry.AreaDefinition(area_id, proj_id, description, proj_dict, width, height, area_extent)
    return area_def


def create_other_area():
    area_id = 'ease_sh'
    description = 'Antarctic EASE grid'
    proj_id = 'ease_sh'
    projection = {'proj': 'longlat', "ellps": "WGS84", "datum": "WGS84"}

    llx = -10  # lower left x coordinate in degrees
    lly = 35.9  # lower left y coordinate in degrees
    urx = 3.5  # upper right x coordinate in degrees
    ury = 55.9  # upper right y coordinate in degrees
    resolution = 0.005

    width = int((urx - llx) / resolution)
    height = int((ury - lly) / resolution)
    area_extent = (llx, lly, urx, ury)

    area_def = pr.geometry.AreaDefinition(area_id, description, proj_id, projection, width, height, area_extent)
    return area_def


if __name__ == '__main__':
    # pol =  Location(50, 10, 'Etc/GMT-1', 34, 'Berlin')
    # times = pandas.date_range(start='2020-04-09', end='2020-04-10', freq='15min')
    # cs = pol.get_clearsky(times)
    # pprint(type(cs))
    # cs.plot()
    # plt.ylabel('Irradiance $W/m^2$')
    # plt.title('Ineichen, climatological turbidity')
    # plt.show()
    # Global Horizontal (GHI) = Direct Normal (DNI) X cos(θ) + Diffuse Horizontal (DHI)
    file = './data/MSG4-SEVI-MSG15-0100-NA-20210409001242.762000000Z-20210409001300-1496632.nat'
    reader = "seviri_l1b_native"
    scene = Scene(filenames={"seviri_l1b_native": [file]})
    read_seviri_data(scene, file, reader, "radiance", "WV_073")
    area = create_other_area()
    #resample(scene, "HRV", area)
