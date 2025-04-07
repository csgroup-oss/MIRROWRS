import numpy as np
import pyproj
import pytest
from eo4hydymogene.gis import project


def test_sig_project_1():
    """Test the project function"""

    lon = np.array([43.600000])
    lat = np.array([1.433333])
    x_tst, y_tst = project(lon, lat)

    assert x_tst == 0.0
    assert y_tst == 0.0


def test_sig_inverse_project_1():
    """ """

    lon_gold = np.array([43.600000])
    lat_gold = np.array([1.433333])

    x = np.array([0.0])
    y = np.array([0.0])

    lon_tst, lat_tst = project(
        x, y, lon_0=lon_gold[0], lat_0=lat_gold[0], x_0=0.0, y_0=0.0, inverse=True
    )
    assert lon_tst == lon_gold
    assert lat_tst == lat_gold


def test_sig_project_2():
    """ """

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


def test_sig_inverse_project_2():
    """ """

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
