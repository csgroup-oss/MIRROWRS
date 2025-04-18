"""
Dummy conftest.py for mirrowrs.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

import os
from pathlib import Path
import pytest

import numpy as np
from shapely.geometry import LineString, Point
import geopandas as gpd
import pandas as pd
from pyproj import CRS

@pytest.fixture
def bbox_gold_4326():
    # -3.17507, 35.71716, -3.16120, 35.72684
    return -3.17501, 35.71712, -3.16114, 35.72680

@pytest.fixture
def bbox_gold_2154():
    # 133000.0, 5419000.0, 134200.0, 5420000.0
    return 133005.0, 5418995.0, 134205.0, 5419995.0

@pytest.fixture
def dpath_inputs():
    return Path(__file__).parent / "inputs"

@pytest.fixture
def dpath_outputs():
    return Path(__file__).parent / "outputs"

@pytest.fixture
def fpath_wm_base_large():
    return str(Path(__file__).parent / "inputs/wm_tus.tif")

@pytest.fixture
def fpath_wm_base_small():
    return str(Path(__file__).parent / "inputs/wm_small_tus.tif")

@pytest.fixture
def fpath_wm_base_small_double():
    return str(Path(__file__).parent / "inputs/wm_small_tus_double.tif")

@pytest.fixture
def fpath_wm_small_exclude():
    return str(Path(__file__).parent / "inputs/wm_small_tus_additionalvalue.tif")

@pytest.fixture
def fpath_wm_small_islands():
    return str(Path(__file__).parent / "inputs/wm_small_tus_island.tif")

@pytest.fixture
def fpath_wm_small_islands_shape_fine():
    return str(Path(__file__).parent / "inputs/wm_small_tus_islands.shp")

@pytest.fixture
def fpath_wm_small_islands_shape_coarse():
    return str(Path(__file__).parent / "inputs/wm_small_tus_islands_full.shp")

@pytest.fixture
def fpath_reaches_large():
    return str(Path(__file__).parent / "inputs/reaches_tus.shp")

@pytest.fixture
def gdf_reaches_large_gold():
    int_epsg = 2154
    lin_1 = LineString([(133600., 5419000.), (133600., 5419500.)])
    lin_2 = LineString([(133600., 5419500.), (133600., 5420000.)])
    gdf_reaches = gpd.GeoDataFrame(
        pd.DataFrame({"reach_id": ["100", "101"]}),
        geometry=gpd.GeoSeries([lin_1, lin_2], crs=CRS(int_epsg)),
        crs=CRS(int_epsg)
    )

    return gdf_reaches

@pytest.fixture
def fpath_nodes_large():
    return str(Path(__file__).parent / "inputs/nodes_tus.shp")

@pytest.fixture
def gdf_nodes_large_gold():
    int_epsg = 2154
    npar_float_node_lat = np.arange(start=5420000. - 125., step=-250., stop=5419000.)
    npar_float_node_lon = 133600. * np.ones_like(npar_float_node_lat)
    df_nodes = pd.DataFrame({"lon": npar_float_node_lon, "lat": npar_float_node_lat})
    df_nodes["node_id"] = ["10101", "10100", "10001", "10000"]
    df_nodes["reach_id"] = ["101", "101", "100", "100"]
    df_nodes["width"] = [100., 100., 100., 100.]

    gdf_nodes = gpd.GeoDataFrame(
        df_nodes,
        geometry=gpd.GeoSeries.from_xy(npar_float_node_lon, npar_float_node_lat, crs=CRS(int_epsg)),
        crs=CRS(int_epsg)
    )

    return gdf_nodes

@pytest.fixture
def fpath_sections_large():
    return str(Path(__file__).parent / "inputs/sections_tus.shp")

@pytest.fixture
def gdf_sections_large_gold():

    int_epsg = 2154
    npar_float_node_lat = np.arange(start=5420000. - 125., step=-250., stop=5419000.)
    npar_float_node_lon = 133600. * np.ones_like(npar_float_node_lat)
    df_nodes = pd.DataFrame({"lon": npar_float_node_lon, "lat": npar_float_node_lat})
    df_nodes["node_id"] = ["10101", "10100", "10001", "10000"]
    df_nodes["reach_id"] = ["101", "101", "100", "100"]
    df_nodes["width_prd"] = [100., 100., 100., 100.]

    l_sections = []
    for lat in npar_float_node_lat:
        lin_section = LineString([(133600. - 250., lat), (133600. + 250., lat)])
        l_sections.append(lin_section)

    gdf_sections = gpd.GeoDataFrame(
        df_nodes,
        geometry=gpd.GeoSeries(l_sections, crs=CRS(int_epsg)),
        crs=CRS(int_epsg)
    )

    return gdf_sections

@pytest.fixture
def flt_width_factor():
    return 5.

@pytest.fixture
def fpath_buffers_large():
    return str(Path(__file__).parent / "inputs/buffer_tus.shp")

@pytest.fixture
def gser_buffers_large_gold():
    int_epsg = 2154
    npar_float_node_lat = np.arange(start=5420000. - 125., step=-250., stop=5419000.)

    l_sections = []
    for lat in npar_float_node_lat:
        lin_section = LineString([(133600. - 250., lat), (133600. + 250., lat)])
        l_sections.append(lin_section)

    gser_sections = gpd.GeoSeries(l_sections, crs=CRS(int_epsg))
    gser_buffers = gser_sections.buffer(distance=40., cap_style="flat")

    return gser_buffers

@pytest.fixture
def fpath_buffers_short():
    return str(Path(__file__).parent / "inputs/buffer_short_tus.shp")

@pytest.fixture
def gser_buffers_short_gold():
    int_epsg = 2154
    npar_float_node_lat = np.arange(start=5420000. - 125., step=-250., stop=5419000.)

    l_sections = []
    for lat in npar_float_node_lat:
        lin_section = LineString([(133600. - 50., lat), (133600. + 50., lat)])
        l_sections.append(lin_section)

    gser_sections = gpd.GeoSeries(l_sections, crs=CRS(int_epsg))
    gser_buffers = gser_sections.buffer(distance=40., cap_style="flat")

    return gser_buffers

@pytest.fixture
def buffer_length():
    return 80.

@pytest.fixture
def gdf_waterbuffer_gold():
    fpath = str(Path(__file__).parent / "inputs/waterbuffer_beta.shp")
    return gpd.read_file(fpath)
