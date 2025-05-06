# Copyright (C) 2024-2025 CS GROUP, https://csgroup.eu
#
# This file is part of MIRROWRS (Mapper to InfeR River Observations of Widths from Remote Sensing)
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
module test_watermask.py
: Unit tests for module mirrowrs/watermask.py
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS
from shapely.geometry import Point

from mirrowrs.tools import DimensionError, FileExtensionError
from mirrowrs.watermask import WaterMask, exclude_value_from_flattened_band

@pytest.fixture
def gold_wm_small_pixc():
    """
    GeoDataFrame containing the pixel-cloud version of the test watermask wm_small_tus.tif
    :return: gpd.GeoDataFrame
    """

    int_height = 10
    int_width = 12
    flt_min_lon = 133000 + 5.
    flt_max_lat = 5420000 - 5.
    flt_resolution = 10.0
    int_epsg = 2154

    npar_int_band_gold = 255 * np.ones((int_height, int_width), dtype=np.uint8)
    npar_int_band_gold[:, 4:8] = 1

    npar_int_band_flat_gold = npar_int_band_gold.flatten()
    indices_gold = exclude_value_from_flattened_band(
        npar_band_flat=npar_int_band_flat_gold, value_to_exclude=255
    )

    df_gold = pd.DataFrame(index=indices_gold, columns=["i", "j", "label", "clean"])
    df_gold["i"] = [i // int_width for i in indices_gold]
    df_gold["j"] = [i % int_width for i in indices_gold]
    df_gold["label"] = 1
    df_gold["clean"] = 1
    for col_name in ["i", "j"]:
        df_gold[col_name] = df_gold[col_name].astype(np.int64)
    for col_name in ["label", "clean"]:
        df_gold[col_name] = df_gold[col_name].astype(np.uint8)

    npar_float_x = np.arange(
        start=flt_min_lon + 0.5 * flt_resolution,
        stop=flt_min_lon + 0.5 * flt_resolution + flt_resolution * int_width,
        step=flt_resolution,
    )
    npar_float_y = np.arange(
        start=flt_max_lat - 0.5 * flt_resolution,
        stop=flt_max_lat - 0.5 * flt_resolution - flt_resolution * int_height,
        step=-flt_resolution,
    )
    gser_gold = gpd.GeoSeries(
        [
            Point(npar_float_x[j], npar_float_y[i])
            for (i, j) in zip(df_gold["i"], df_gold["j"])
        ],
        index=df_gold.index,
        crs=CRS(int_epsg),
    )
    gdf_band_as_pixc_gold = gpd.GeoDataFrame(
        df_gold, geometry=gser_gold, crs=CRS(int_epsg)
    )

    return gdf_band_as_pixc_gold


# Test WaterMask() instantiation
def test_init_basic():
    """
    Check if the WaterMask class is correctly instanciated
    """
    obj = WaterMask()
    assert isinstance(obj, WaterMask)


# Test WaterMask() instantiation : check if default values are set correctly
def test_init_default_values():
    """
    When instanciating a WaterMask object, check if default values are set correctly
    """
    obj = WaterMask()

    assert obj.str_provider is None
    assert obj.str_fpath_infile is None
    assert obj.bbox is None
    assert obj.crs is None
    assert obj.crs_epsg is None
    assert obj.coordsyst is None
    assert obj.width is None
    assert obj.height is None
    assert obj.dtypes is None
    assert obj.nodata is None
    assert obj.res is None
    assert obj.gdf_wm_as_pixc is None
    assert obj.dtype_label_out is None


# Test @classmethod : return right object
def test_from_tif_creates_instance(fpath_wm_base_large):
    """
    Check if method class decorator return the right object
    """
    obj = WaterMask.from_tif(
        watermask_tif=fpath_wm_base_large,
        str_origin="my_source",
        str_proj="proj"
    )
    assert isinstance(obj, WaterMask)
    assert obj.str_provider == "my_source"
    assert obj.coordsyst == "proj"


# Test @classmethod : check attribute values
def test_init_attributes(fpath_wm_base_large, bbox_gold_2154):
    """
    Test @classmethod : check attribute values
    """
    obj = WaterMask.from_tif(
        watermask_tif=fpath_wm_base_large, str_origin="my_source", str_proj="proj"
    )

    assert obj.str_provider == "my_source"
    assert obj.str_fpath_infile == fpath_wm_base_large
    assert obj.coordsyst == "proj"

    assert obj.bbox == bbox_gold_2154
    assert obj.crs == "EPSG:2154"
    assert obj.crs_epsg == 2154
    assert obj.width == 120
    assert obj.height == 100
    assert obj.dtypes == rasterio.uint8
    assert obj.nodata == 255
    assert obj.res == 10.0
    assert obj.dtype_label_out == rasterio.uint8


# Test @classmethod : check if wrong inputs raise right Exception
@pytest.mark.parametrize(
    "wm_in, origin, proj, expected_error",
    [
        ("wm_tus.tif", 123, "proj", TypeError),
        ("not_wm_tus.tif", "my_source", "proj", FileExistsError),
        ("wm_tus.shp", "my_source", "proj", FileExtensionError),
        ("wm_tus.tif", "my_source", "other", NotImplementedError),
    ],
)
def test_from_tif_wrong_inputs(dpath_inputs, wm_in, origin, proj, expected_error):
    """Test @classmethod : check if wrong inputs raise right Exception
    """
    fpath_wm_in = os.path.join(dpath_inputs, wm_in)
    with pytest.raises(expected_error):
        WaterMask.from_tif(watermask_tif=fpath_wm_in, str_origin=origin, str_proj=proj)


# Test __str__ method in WaterMask class
def test_str_method(fpath_wm_base_large):
    """
    Test __str__ method in WaterMask class
    """

    obj = WaterMask()
    assert str(obj) == "Empty WaterMask."

    obj = WaterMask.from_tif(
        watermask_tif=fpath_wm_base_large, str_origin="my_source", str_proj="proj"
    )
    assert str(obj) == "WaterMask product from my_source."

    obj = WaterMask.from_tif(watermask_tif=fpath_wm_base_large, str_proj="proj")

    assert "WaterMask product from" in str(obj)
    assert "inputs/wm_tus.tif." in str(obj)


# Test method get_bbox #1
def test_get_bbox_1(fpath_wm_base_large, bbox_gold_4326):
    """Test method get_bbox
    """

    wm_tst = WaterMask.from_tif(fpath_wm_base_large)
    bbox_tst = wm_tst.get_bbox()

    assert bbox_tst[0] == pytest.approx(bbox_gold_4326[0], abs=1e-5)
    assert bbox_tst[1] == pytest.approx(bbox_gold_4326[1], abs=1e-5)
    assert bbox_tst[2] == pytest.approx(bbox_gold_4326[2], abs=1e-5)
    assert bbox_tst[3] == pytest.approx(bbox_gold_4326[3], abs=1e-5)

# Test method get_bbox #2
def test_get_bbox_2(fpath_wm_base_large, bbox_gold_2154):
    """Test method get_bbox
    """

    # Cheat to access all parts of if in method
    wm_tst = WaterMask.from_tif(fpath_wm_base_large, str_proj="lonlat")
    bbox_tst = wm_tst.get_bbox()

    assert bbox_tst[0] == pytest.approx(bbox_gold_2154[0], abs=1e-5)
    assert bbox_tst[1] == pytest.approx(bbox_gold_2154[1], abs=1e-5)
    assert bbox_tst[2] == pytest.approx(bbox_gold_2154[2], abs=1e-5)
    assert bbox_tst[3] == pytest.approx(bbox_gold_2154[3], abs=1e-5)


# Test exclude_value_from_flattened_band function : check if wrong inputs raise right Exception
@pytest.mark.parametrize(
    "band_in, value_to_exclude, expected_error",
    [
        (1, 0.0, TypeError),
        (np.ones((2, 2)), 0.0, DimensionError),
        (np.ones((4,)), "a", ValueError),
    ],
)
def test_exclude_value_from_flattened_band_wrong_input_type(
    band_in, value_to_exclude, expected_error
):
    """
    Test exclude_value_from_flattened_band function : check if wrong inputs raise right Exception
    """
    with pytest.raises(expected_error):
        _ = exclude_value_from_flattened_band(
            npar_band_flat=band_in, value_to_exclude=value_to_exclude
        )


# Test exclude_value_from_flattened_band function
@pytest.mark.parametrize("excluded_value", [0.0, np.nan, np.inf])
def test_exclude_value_from_flattened_band(excluded_value):
    """
    Test exclude_value_from_flattened_band function
    """

    npar_input = np.ones((4,))
    npar_input[0] = excluded_value

    indices_test = exclude_value_from_flattened_band(
        npar_band_flat=npar_input, value_to_exclude=excluded_value
    )
    indices_gold = np.array([1, 2, 3], dtype=np.int64)

    assert np.array_equal(indices_test, indices_gold)


# Test @staticmethod from band_to_pixc - exclude_values=None
def test_band_to_pixc_without_excluded_value(gold_wm_small_pixc, fpath_wm_base_small):
    """
    Test @staticmethod from band_to_pixc - exclude_values=None
    """

    with rasterio.open(fpath_wm_base_small, "r") as raster_src:
        gdf_band_as_pixc_test = WaterMask.band_to_pixc(raster_src)

    assert_geodataframe_equal(gdf_band_as_pixc_test, gold_wm_small_pixc)


# Test @staticmethod from band_to_pixc - exclude_values=2
def test_band_to_pixc_with_excluded_value(fpath_wm_small_exclude, gold_wm_small_pixc):
    """
    Test @staticmethod from band_to_pixc - exclude_values=2
    """

    # Gold variables
    # npar_int_band_gold[0:2, 4:6] = 255
    gold_wm_small_pixc.drop(labels=[4, 5, 16, 17], axis=0, inplace=True)

    with rasterio.open(fpath_wm_small_exclude, "r") as raster_src:
        gdf_band_as_pixc_test = WaterMask.band_to_pixc(raster_src, exclude_values=2)

    assert_geodataframe_equal(gdf_band_as_pixc_test, gold_wm_small_pixc)


# Test @staticmethod from band_to_pixc - check if wrong inputs raise right Exception - NotImplementedError
def test_band_to_pixc_iterable_excluded_value(fpath_wm_base_small):
    """
    Test @staticmethod from band_to_pixc - check if wrong inputs raise right Exception - NotImplementedError
    """
    with pytest.raises(NotImplementedError):
        with rasterio.open(fpath_wm_base_small, "r") as raster_src:
            _ = WaterMask.band_to_pixc(raster_src, exclude_values=[0, 1])


# Test @staticmethod from band_to_pixc - check if warning is correctly raised
def test_band_to_pixc_warning(fpath_wm_base_small_double, caplog):
    """Check if warning is correctly raised
    """
    with caplog.at_level("WARNING"):
        with rasterio.open(fpath_wm_base_small_double, "r") as raster_src:
            _ = WaterMask.band_to_pixc(raster_src)
    assert "More than 1 band in the rasterio dataset, use only first one." in caplog.text

# Test method update_clean_flag: check if wrong inputs raise right Exception
@pytest.mark.parametrize(
    "mask_clean, expected_exception",
    [(0.0, TypeError), ([4, "a"], TypeError), ([0], ValueError)],
)
def test_update_clean_flag_wrong_inputs(mask_clean, expected_exception, fpath_wm_base_small):
    """
    Test method update_clean_flag: check if wrong inputs raise right Exception
    """

    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    with pytest.raises(expected_exception):
        wm_tst.update_clean_flag(mask=mask_clean)


# Test method update_clean_flag
def test_update_clean_flag(gold_wm_small_pixc, fpath_wm_base_small):
    """
    Test method update_clean_flag
    """

    # Gold variable
    # df_gold.loc[[4,5], "clean"] = 0
    gold_wm_small_pixc.loc[[4, 5], "clean"] = 0

    # Test variable
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    wm_tst.update_clean_flag(mask=[4, 5])

    assert_geodataframe_equal(wm_tst.gdf_wm_as_pixc, gold_wm_small_pixc)


# Test method update_label_flag: check if wrong inputs raise right Exception
@pytest.mark.parametrize(
    "dct_label, expected_exception",
    [
        ({2: 0.0}, TypeError),
        ({2: [4, "a"]}, TypeError),
        ({2: [0]}, ValueError),
        ({"a": [4, 5]}, ValueError),
        ("a", TypeError),
        ({66000: [4, 5]}, NotImplementedError),
    ],
)
def test_update_label_flag_wrong_inputs(dct_label, expected_exception, fpath_wm_base_small):
    """
    Test method update_label_flag: check if wrong inputs raise right Exception
    """

    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    with pytest.raises(expected_exception):
        wm_tst.update_label_flag(dct_label=dct_label)

# Test method update_label_flag: check if warning is raised
def test_update_label_flag_warning(fpath_wm_base_small, caplog):
    """Check if warning is raised
    """
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)

    with caplog.at_level("WARNING"):
        wm_tst.update_label_flag(dct_label={2.5: [4,5,6,7]})
    assert "Label is not an integer, will be changed to integer counterpart" in caplog.text

# Test method update_label_flag
@pytest.mark.parametrize(
    "label, dtype_out, val_nodata",
    [(2, rasterio.uint8, 255), (500, rasterio.uint16, 65535)],
)
def test_update_label_flag(fpath_wm_base_small, gold_wm_small_pixc, label, dtype_out, val_nodata):
    """
    Test method update_label_flag
    """

    # Gold variables
    # Update label for test
    if label > 255:
        gold_wm_small_pixc["label"] = gold_wm_small_pixc["label"].astype(np.uint16)
        gold_wm_small_pixc.loc[[4, 5, 6, 7], "label"] = label
    else:
        gold_wm_small_pixc.loc[[4, 5, 6, 7], "label"] = label

    # Test variable
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    wm_tst.update_label_flag(dct_label={label: [4, 5, 6, 7]})

    # Assert method
    assert_geodataframe_equal(wm_tst.gdf_wm_as_pixc, gold_wm_small_pixc)
    assert wm_tst.dtype_label_out == dtype_out
    assert wm_tst.nodata == val_nodata


# Test method get_band
@pytest.mark.parametrize(
    "bool_clean, bool_label, bool_masked_array",
    [
        (True, True, False),
        (True, False, False),
        (False, True, False),
        (False, False, False),
        (True, True, True),
    ],
)
def test_get_band(fpath_wm_base_small, bool_clean, bool_label, bool_masked_array):
    """
    Test method get_band
    """

    # Gold variables
    int_height = 10
    int_width = 12
    npar_int_band_gold = 255 * np.ones((int_height, int_width), dtype=np.uint8)
    npar_int_band_gold[:, 4:8] = 1

    # Add label
    if bool_label:
        npar_int_band_gold[:5, 4:8] = 2
        npar_int_band_gold[5:, 4:8] = 3
    if bool_clean:
        npar_int_band_gold[0, 4:8] = 255
    if bool_masked_array:
        npar_int_band_gold = np.ma.array(
            npar_int_band_gold,
            mask=(npar_int_band_gold == 255),
        )

    # Test variable
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    wm_tst.update_clean_flag(mask=[4, 5, 6, 7])
    wm_tst.update_label_flag(
        dct_label={
            2: [
                4,
                5,
                6,
                7,
                16,
                17,
                18,
                19,
                28,
                29,
                30,
                31,
                40,
                41,
                42,
                43,
                52,
                53,
                54,
                55,
            ],
            3: [
                64,
                65,
                66,
                67,
                76,
                77,
                78,
                79,
                88,
                89,
                90,
                91,
                100,
                101,
                102,
                103,
                112,
                113,
                114,
                115,
            ],
        }
    )

    npar_int_band_tst = wm_tst.get_band(bool_clean, bool_label, bool_masked_array)

    # Assert method
    if not bool_masked_array:
        np.array_equal(npar_int_band_tst, npar_int_band_gold)
    else:
        np.ma.allequal(npar_int_band_tst, npar_int_band_gold)


# Test method get_polygons
@pytest.mark.parametrize(
    "bool_exterior_only, bool_indices",
    [(True, False), (False, False), (True, True), (True, False)],
)
def test_get_polygons(fpath_wm_small_islands,
                      fpath_wm_small_islands_shape_fine,
                      fpath_wm_small_islands_shape_coarse,
                      bool_exterior_only,
                      bool_indices):
    """
    Test method get_polygons
    """

    # Gold variable
    int_height = 10
    int_width = 12
    npar_int_band_gold = 255 * np.ones((int_height, int_width), dtype=np.uint8)

    npar_int_band_gold[:, 4:8] = 1
    npar_int_band_gold[4:8, 5:7] = 255
    indices = list(np.where(npar_int_band_gold.flatten() != 255))

    if bool_exterior_only:
        str_fpath_wm_polygons_shp = fpath_wm_small_islands_shape_coarse
    else:
        str_fpath_wm_polygons_shp = fpath_wm_small_islands_shape_fine

    if not bool_indices:
        indices = None

    gdf_tmp_gold = gpd.read_file(str_fpath_wm_polygons_shp)
    gser_gold = gdf_tmp_gold["geometry"]
    del gdf_tmp_gold

    gdf_wm_polygons_gold = gpd.GeoDataFrame(
        pd.DataFrame({"label": [1], "clean": [1], "indices": None}),
        geometry=gser_gold,
        crs=gser_gold.crs,
    )
    gdf_wm_polygons_gold["label"] = gdf_wm_polygons_gold["label"].astype(np.uint8)
    gdf_wm_polygons_gold["clean"] = gdf_wm_polygons_gold["clean"].astype(np.uint8)
    gdf_wm_polygons_gold["indices"] = indices

    # Test variable
    wm_tst = WaterMask.from_tif(fpath_wm_small_islands)
    gdf_wm_polygons_tst = wm_tst.get_polygons(
        bool_clean=False,
        bool_label=False,
        bool_exterior_only=bool_exterior_only,
        bool_indices=bool_indices,
    )

    assert_geodataframe_equal(gdf_wm_polygons_tst, gdf_wm_polygons_gold)


# Test method save_wm : check if wrong inputs raise right Exception
def test_save_wm_wrong_inputs(fpath_wm_base_small):
    """
    Test method save_wm : check if wrong inputs raise right Exception
    """
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    with pytest.raises(NotImplementedError):
        wm_tst.save_wm(fmt="abc")


# Test method save_wm_as_tif : check if wrong inputs raise right Exception
@pytest.mark.parametrize(
    "bool_clean, bool_label, output_dir, expected_error",
    [
        ("a", True, ".", TypeError),
        (True, "a", ".", TypeError),
        (True, True, "./not_a_dir", NotADirectoryError),
    ],
)
def test_save_wm_as_tif_wrong_inputs(
    fpath_wm_base_small, bool_clean, bool_label, output_dir, expected_error
):
    """
    Test method save_wm_as_tif : check if wrong inputs raise right Exception
    """
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    with pytest.raises(expected_error):
        wm_tst.save_wm_as_tif(
            bool_clean=bool_clean, bool_label=bool_label, str_fpath_dir_out=output_dir
        )


# Test method save_wm_as_tif : check if output filename is correct
@pytest.mark.parametrize(
    "bool_clean, bool_label, str_suffix, expected_out",
    [
        (True, False, "suffix-1", "wm_small_tus_clean_suffix-1.tif"),
        (False, True, "suffix-2", "wm_small_tus_label_suffix-2.tif"),
        (False, False, "suffix-3", "wm_small_tus_suffix-3.tif"),
        (True, True, "suffix-4", "wm_small_tus_clean_label_suffix-4.tif"),
        (False, False, None, "wm_small_tus.tif"),
    ],
)
def test_save_wm_as_tif_out_filename(dpath_outputs, fpath_wm_base_small, bool_clean, bool_label, str_suffix, expected_out):
    """
    Test method save_wm_as_tif : check if output filename is correct
    """
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    str_wm_test_out = wm_tst.save_wm_as_tif(
        bool_clean=bool_clean,
        bool_label=bool_label,
        str_fpath_dir_out=dpath_outputs,
        str_suffix=str_suffix,
    )
    assert str_wm_test_out == os.path.join(dpath_outputs, expected_out)


# Test method save_wm_as_tif : file is correctly created
def test_save_wm_as_tif_created(dpath_outputs, fpath_wm_base_small):
    """
    Test method save_wm_as_tif : file is correctly created
    """
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    str_tif_out_tst = wm_tst.save_wm_as_tif(str_fpath_dir_out=dpath_outputs)

    assert os.path.isfile(str_tif_out_tst)


# Test method save_wm_as_tif : file content is correct
def test_save_wm_as_tif_content(dpath_outputs, fpath_wm_base_small):
    """
    Test method save_wm_as_tif : file content is correct
    """
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    npar_band_gold = wm_tst.get_band()

    str_tif_out_tst = wm_tst.save_wm_as_tif(str_fpath_dir_out=dpath_outputs)
    with rasterio.open(str_tif_out_tst) as src:
        npar_band_test = src.read(1)

    assert np.array_equal(npar_band_test, npar_band_gold)


# Test method save_wm_as_tif : file metadata is correct
def test_save_wm_as_tif_metadata(dpath_outputs, fpath_wm_base_small):
    """
    Test method save_wm_as_tif : file metadata is correct
    """
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    str_tif_out_tst = wm_tst.save_wm_as_tif(str_fpath_dir_out=dpath_outputs)
    with rasterio.open(str_tif_out_tst) as src:
        assert src.crs == wm_tst.crs
        assert src.transform == wm_tst.transform
        assert src.count == 1


# Test method save_wm_as_pixc : check if wrong inputs raise right Exception
def test_save_wm_as_pixc_wrong_inputs(fpath_wm_base_small):
    """
    Test method save_wm_as_pixc : check if wrong inputs raise right Exception
    """
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    with pytest.raises(NotADirectoryError):
        _ = wm_tst.save_wm_as_pixc(str_fpath_dir_out="./not_a_directory")


# Test method save_wm_as_pixc : check if output filename is correct
def test_save_wm_as_pixc_out_filename(dpath_outputs, fpath_wm_base_small):
    """
    Test method save_wm_as_pixc : check if output filename is correct
    """
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    str_fpath_out_tst = wm_tst.save_wm_as_pixc(str_fpath_dir_out=dpath_outputs)

    assert str_fpath_out_tst == os.path.join(dpath_outputs, "wm_small_tus_pixc.shp")


# Test method save_wm_as_pixc : file is correctly created
def test_save_wm_as_pixc_created(dpath_outputs, fpath_wm_base_small):
    """
    Test method save_wm_as_pixc : file is correctly created
    """
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    str_fpath_out_tst = wm_tst.save_wm_as_pixc(str_fpath_dir_out=dpath_outputs)

    assert os.path.isfile(str_fpath_out_tst)
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        assert os.path.isfile(os.path.join(dpath_outputs,"wm_small_tus_pixc" + ext))


# Test method save_wm_as_pixc : data integrity
def test_save_wp_as_pixc_data_integrity(dpath_outputs, fpath_wm_base_small):
    """
    Test method save_wm_as_pixc : data integrity
    """
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    gdf_gold = wm_tst.gdf_wm_as_pixc.reset_index(drop=False, inplace=False)

    str_fpath_out_tst = wm_tst.save_wm_as_pixc(str_fpath_dir_out=dpath_outputs)
    gdf_tst = gpd.read_file(str_fpath_out_tst)

    pd.testing.assert_frame_equal(
        gdf_gold.sort_index(axis=1), gdf_tst.sort_index(axis=1), check_dtype=False
    )


# Test method save_wm_as_shp : check if wrong inputs raise right Exception
@pytest.mark.parametrize(
    "bool_clean, bool_label, output_dir, expected_error",
    [
        ("a", True, ".", TypeError),
        (True, "a", ".", TypeError),
        (True, True, "./not_a_dir", NotADirectoryError),
    ],
)
def test_save_wm_as_shp_wrong_inputs(
    fpath_wm_base_small, bool_clean, bool_label, output_dir, expected_error
):
    """
    Test method save_wm_as_shp : check if wrong inputs raise right Exception
    """
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    with pytest.raises(expected_error):
        wm_tst.save_wm_as_shp(
            bool_clean=bool_clean, bool_label=bool_label, str_fpath_dir_out=output_dir
        )


# Test method save_wm_as_tif : check if output filename is correct
@pytest.mark.parametrize(
    "bool_clean, bool_label, str_suffix, expected_out",
    [
        (True, False, "suffix-1", "wm_small_tus_clean_suffix-1.shp"),
        (False, True, "suffix-2", "wm_small_tus_label_suffix-2.shp"),
        (False, False, "suffix-3", "wm_small_tus_suffix-3.shp"),
        (True, True, "suffix-4", "wm_small_tus_clean_label_suffix-4.shp"),
        (False, False, None, "wm_small_tus.shp"),
    ],
)
def test_save_wm_as_shp_out_filename(dpath_outputs, fpath_wm_base_small, bool_clean, bool_label, str_suffix, expected_out):
    """
    Test method save_wm_as_tif : check if output filename is correct
    """
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    str_wm_test_out = wm_tst.save_wm_as_shp(
        bool_clean=bool_clean,
        bool_label=bool_label,
        str_fpath_dir_out=dpath_outputs,
        str_suffix=str_suffix,
    )
    assert str_wm_test_out == os.path.join(dpath_outputs, expected_out)


# Test method save_wm_as_shp : file is correctly created
def test_save_wm_as_shp_created(dpath_outputs, fpath_wm_base_small):
    """
    Test method save_wm_as_shp : file is correctly created
    """
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    _ = wm_tst.save_wm_as_shp(
        bool_clean=False, bool_label=False, str_fpath_dir_out=dpath_outputs
    )
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        assert os.path.isfile(os.path.join(dpath_outputs,"wm_small_tus_pixc" + ext))


# Test method save_wm_as_shp : data integrity
def test_save_wm_as_shp_data_integrity(dpath_outputs, fpath_wm_base_small):
    """
    Test method save_wm_as_shp : data integrity
    """
    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    gdf_gold = wm_tst.get_polygons()
    gdf_gold["indices"] = gdf_gold["indices"].apply(str)

    str_fpath_out_tst = wm_tst.save_wm_as_shp(str_fpath_dir_out=dpath_outputs)
    gdf_tst = gpd.read_file(str_fpath_out_tst)

    assert_geodataframe_equal(gdf_tst, gdf_gold, check_dtype=False)

# Test behavior of method save_wm
@pytest.mark.parametrize("extension, method, output_file",
[("tif", "save_wm_as_tif", "wm_small_tus.tif"),
 ("pixc", "save_wm_as_pixc", "wm_small_tus_pixc.shp"),
 ("shp", "save_wm_as_shp", "wm_small_tus.shp")],)
def test_save_wm(mocker, fpath_wm_base_small, dpath_outputs, extension, method, output_file):
    """Test behavior of method save_wm
    """

    wm_tst = WaterMask.from_tif(fpath_wm_base_small)
    mocker_save_tif = mocker.patch.object(wm_tst, method, return_value=os.path.join(dpath_outputs, output_file))

    fapth_out_tst = wm_tst.save_wm(fmt=extension,
                   bool_clean=False,
                   bool_label=False,
                   str_fpath_dir_out=dpath_outputs)

    mocker_save_tif.assert_called_once()
    assert fapth_out_tst == os.path.join(dpath_outputs, output_file)



