# Copyright (C) 2024-2025 CS GROUP, https://csgroup.eu
#
# This file is part of MIRROWRS (Earth Observations For HydrauDynamic Model Generation)
#
#     https://github.com/csgroup-oss/MIRROWRS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
sig.py
: module with miscellaneous functions related to SIG manipulation
"""

import logging

import numpy as np
import pyproj

_logger = logging.getLogger("gis_module")

# Function to project WGS84 data to "laea" projection
def project(
    lon,
    lat,
    proj="laea",
    x_0=0,
    y_0=0,
    lat_0=None,
    lon_0=None,
    ellps="WGS84",
    inverse=False,
    **proj_kwds,
):
    """project 2D-coordinates from WGS84 (lon-lat system) centered around lon-lat domain
    to "laea" (x-y system) system (by default centered on (x_O=0.,y_0=0.))
    or inverse if inverse=True

    :param lon: np.ndarray like
        longitude array
    :param lat: np.ndarray-like
        latitude array
    :param proj: str
        projected coordinate system to convert to
        by-default "laea"
    :param x_0: float
        Geometric longitude of projection center
    :param y_0: float
        Geometric latitude of projection center
    :param lat_0: float
        Latitude of projection center
    :param lon_0: float
        Longitude of projection center
    :param ellps: str
        Geographic system reference ellipsoid
    :param inverse: boolean
        If True, perform projection transformation from (x,y) system to (lon,lat) system
    :param proj_kwds:
    :return x: np.ndarray-like with lon.shape
        projected x-coordinate array
    :return y: np.ndarray-like with lat.shape
        projected y-coordinate array
    """

    if lat_0 is None:
        lat_0 = (np.amax(lat) + np.amin(lat)) / 2.0

    if lon_0 is None:
        lon_0 = (np.amin(lon) + np.amax(lon)) / 2.0

    proj = pyproj.Proj(
        proj=proj, lat_0=lat_0, lon_0=lon_0, x_0=x_0, y_0=y_0, ellps=ellps, **proj_kwds
    )

    if not inverse:
        x, y = proj(lon, lat)
    else:
        x, y = proj(lon, lat, inverse=True)

    return x, y