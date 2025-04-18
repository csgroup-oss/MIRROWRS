# Copyright (C) 2024 CNES.
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
module test_gis.py
: Unit tests for module mirrowrs/gis.py
"""

import numpy as np
import pyproj
from pyproj import CRS
import pytest

from mirrowrs.gis import project, reproject_bbox_to_wgs84

# Test function project #1 = center of domain
def test_gis_project_1():
    """Test the project function"""

    lon = np.array([43.600000])
    lat = np.array([1.433333])
    x_tst, y_tst = project(lon, lat)

    assert x_tst == 0.0
    assert y_tst == 0.0

# Test function project #1 = center of domain, inverse mode
def test_gis_inverse_project_1():
    """Test function project, inverse mode
    """

    lon_gold = np.array([43.600000])
    lat_gold = np.array([1.433333])

    x = np.array([0.0])
    y = np.array([0.0])

    lon_tst, lat_tst = project(
        x, y, lon_0=lon_gold[0], lat_0=lat_gold[0], x_0=0.0, y_0=0.0, inverse=True
    )
    assert lon_tst == lon_gold
    assert lat_tst == lat_gold

# Test function project #2 = any point
def test_gis_project_2():
    """Test the project function
    """

    # Set domain
    flt_min_lon = 1.4
    flt_max_lon = 1.45
    flt_min_lat = 43.55
    flt_max_lat = 43.6

    # Centers of projection
    lat_0 = (flt_max_lat + flt_min_lat) / 2.0
    lon_0 = (flt_max_lon + flt_min_lon) / 2.0
    x_0 = 0
    y_0 = 0

    lon = np.array([43.5625])
    lat = np.array([1.4225])
    x_tst, y_tst = project(lon, lat, lon_0=lon_0, lat_0=lat_0)

    # Ground truth generation
    proj = pyproj.Proj(
        proj="laea", lat_0=lat_0, lon_0=lon_0, x_0=x_0, y_0=y_0, ellps="WGS84"
    )
    x_gold, y_gold = proj(lon, lat)

    assert x_tst == x_gold
    assert y_tst == y_gold

# Test function project #2 = any point, inverse mode
def test_gis_inverse_project_2():
    """Test function project, inverse mode
    """

    lon_gold = np.array([43.5625])
    lat_gold = np.array([1.4225])

    # Set domain
    flt_min_lon = 1.4
    flt_max_lon = 1.45
    flt_min_lat = 43.55
    flt_max_lat = 43.6

    # Centers of projection
    lat_0 = (flt_max_lat + flt_min_lat) / 2.0
    lon_0 = (flt_max_lon + flt_min_lon) / 2.0
    x_0 = 0
    y_0 = 0
    x, y = project(lon_gold, lat_gold, lon_0=lon_0, lat_0=lat_0, x_0=x_0, y_0=y_0)

    # Ground truth generation
    proj = pyproj.Proj(
        proj="laea", lat_0=lat_0, lon_0=lon_0, x_0=x_0, y_0=y_0, ellps="WGS84"
    )
    lon_tst, lat_tst = proj(x, y, inverse=True)

    assert np.abs(lon_tst - lon_gold) <= 0.000001
    assert np.abs(lat_tst - lat_gold) <= 0.000001

# Test if reproject_bbox_to_wgs84 return right outputs
def test_gis_reproject_bbox_to_wgs84(bbox_gold_4326, bbox_gold_2154):
    """Test function: reproject_bbox_to_wgs84
    """

    t_bbox_wsg84 = reproject_bbox_to_wgs84(bbox_gold_2154, CRS(2154))

    assert t_bbox_wsg84[0] == pytest.approx(bbox_gold_4326[0], abs=1e-5)
    assert t_bbox_wsg84[1] == pytest.approx(bbox_gold_4326[1], abs=1e-5)
    assert t_bbox_wsg84[2] == pytest.approx(bbox_gold_4326[2], abs=1e-5)
    assert t_bbox_wsg84[3] == pytest.approx(bbox_gold_4326[3], abs=1e-5)


