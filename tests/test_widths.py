# Copyright (C) 2024-2025 CS GROUP, https://csgroup.eu
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
module test_widths.py
: Unit tests for module mirrowrs/widths.py
"""
import os

from geopandas.testing import assert_geoseries_equal, assert_geodataframe_equal
import geopandas as gpd
import numpy as np
import pandas as pd
import pandas.testing as pd_testing
import pytest
import rasterio as rio
from shapely.geometry import Polygon

from mirrowrs.widths import ParamWidthComp
from mirrowrs.widths import count_pixels
from mirrowrs.widths import quantify_intersection_ratio_between_buffer
from mirrowrs.widths import compute_width_over_one_section
from mirrowrs.widths import compute_widths_from_single_watermask_base

@pytest.fixture
def dct_config_kwargs(dpath_outputs):
    """Return a kwargs-like dictionary to use with the ParamWidthComp class
    :return dct_out: dict
    """

    dct_out = {
        "label_attr" : "label",
        "min_width": 50.,
        "export_buffered_sections": False,
        "bool_print_dry": True,
        "fname_buffered_section": "buffered_sections_tst.shp"
    }

    return dct_out

# Test ParamWidthComp instantiation : with default values
def test_paramwidthcomp_init_default():
    """Test ParamWidthComp instantiation : with default values
    """

    obj = ParamWidthComp
    assert obj.label_attr == ""
    assert obj.min_width == -1
    assert obj.export_buffered_sections is False
    assert obj.bool_print_dry is False
    assert obj.fname_buffered_section == "sections_buffered.shp"

# Test ParamWidthComp instantiation : with specified values
def test_paramwidthcomp_init(dct_config_kwargs):
    """Test ParamWidthComp instantiation : with specified values
    """

    obj = ParamWidthComp(**dct_config_kwargs)
    assert obj.label_attr == "label"
    assert obj.min_width == 50.
    assert obj.export_buffered_sections is False
    assert obj.bool_print_dry is True
    assert obj.fname_buffered_section == "buffered_sections_tst.shp"


# Test ParamWidthComp method __post_init__()
@pytest.mark.parametrize("key, wrong_value, expected_error",
                         [("label_attr", 1, TypeError),
                          ("min_width", "a", TypeError),
                          ("export_buffered_sections", "a", TypeError),
                          ("bool_print_dry", "a", TypeError),
                          ("fname_buffered_section", 1, TypeError),
                          ("fname_buffered_section", "a", ValueError)],)
def test_paramwidthcomp_wrong_inputs(key, wrong_value, expected_error, dct_config_kwargs):
    """Test method ParamWidthComp.__post_init__()
    """
    dct_test = dct_config_kwargs.copy()
    dct_test[key] = wrong_value
    with pytest.raises(expected_error):
        _ = ParamWidthComp(**dct_test)

# Test count_water_pixels : wrong inputs raise right exception
@pytest.mark.parametrize("expected_error, bool_param1, bool_param2, bool_param3, bool_param4, wrong_value",
                         [(ValueError, True, False, False, False, None),
                          (TypeError, True, False, False, False, "a"),
                          (TypeError, False, True, False, False, "a"),
                          (TypeError, False, False, True, False, "a"),
                          (ValueError, False, False, False, True, None),
                          (ValueError, False, False, False, True, "a")])
def test_count_water_pixels_wrong_inputs(expected_error, bool_param1, bool_param2, bool_param3, bool_param4, wrong_value):
    """Test count_water_pixels : wrong inputs raise right exception
    """

    # Set default parameters
    npar_band_tst = 255 * np.ones((10, 16), dtype=np.uint8)
    npar_band_tst[:6,4:12] = 1
    npar_band_tst[6:, 4:12] = 2
    val_nodata_tst = 255
    bool_label_tst = False
    int_label_tst = 1

    # Deactivate parameter to test
    if bool_param1:
        npar_band_tst = wrong_value
    if bool_param2:
        val_nodata_tst = wrong_value
    if bool_param3:
        bool_label_tst = wrong_value
    if bool_param4:
        int_label_tst = wrong_value
        bool_label_tst = True

    # Test
    with pytest.raises(expected_error):
        _ = count_pixels(out_image=npar_band_tst,
                         val_nodata=val_nodata_tst,
                         bool_label=bool_label_tst,
                         int_label=int_label_tst)

# Test function count_water_pixels : check right outputs
@pytest.mark.parametrize("val_nodata",
                         [255,
                          np.nan,
                          np.inf])
def test_count_water_pixels_no_data(val_nodata):
    """Test count_water_pixels : test no-data handling
    """

    # Gold value
    count_gold = 3

    # Set default parameters
    npar_band_tst = val_nodata * np.ones((3,3), dtype=np.uint8)
    npar_band_tst[:,0] = 1
    bool_label_tst = False

    # Test function
    count_tst = count_pixels(out_image=npar_band_tst,
                     val_nodata=val_nodata,
                     bool_label=bool_label_tst)

    assert count_tst == count_gold

@pytest.mark.parametrize("bool_label, expected_value",
                         [(False, 6),
                          (True, 3)])
def test_count_water_pixels_right_outputs(expected_value, bool_label):
    """Test count_water_pixels : return right outputs
    """

    # Gold value
    count_gold = expected_value

    # Set default parameters
    npar_band_tst = 255 * np.ones((3,3), dtype=np.uint8)
    npar_band_tst[:, :2] = 1
    npar_band_tst[:, 0] = 2

    # Test function
    count_tst = count_pixels(out_image=npar_band_tst,
                             val_nodata=255,
                             bool_label=bool_label,
                             int_label=1)

    assert count_tst == count_gold

# Test function compute_width_over_one_section : wrong inputs raise right exception
@pytest.mark.parametrize("expected_error, bool_param1, bool_param2, bool_param3, bool_param4, bool_param5",
                         [(TypeError, True, False, False, False, False),
                          (ValueError, False, True, False, False, False),
                          (ValueError, False, False, True, False, False),
                          (ValueError, False, False, False, True, False),
                          (ValueError, False, False, False, False, True)])
def test_compute_width_over_one_section_wrong_inputs(gser_buffers_large_gold, fpath_wm_base_small, expected_error, bool_param1, bool_param2, bool_param3, bool_param4, bool_param5):
    """Function compute_width_over_one_section : wrong inputs raise right exception
    """

    # Set parameters for test
    pol_section_buffered_tst = gser_buffers_large_gold.loc[0]
    flt_buffer_area_tst = gser_buffers_large_gold.area.loc[0]
    watermask_tst = rio.open(fpath_wm_base_small, "r")
    config_tst = ParamWidthComp
    pixel_area_tst = 100.

    # Deactivate parameter to test
    if bool_param1:
        pol_section_buffered_tst = "a"
    if bool_param2:
        flt_buffer_area_tst = None
    if bool_param3:
        watermask_tst = None
    if bool_param4:
        config_tst = None
    if bool_param5:
        pixel_area_tst = None

    # Test
    with pytest.raises(expected_error):
        _ = compute_width_over_one_section(pol_section_buffered=pol_section_buffered_tst,
                                           flt_buffer_length=25.,
                                           flt_buffer_area=flt_buffer_area_tst,
                                           watermask=watermask_tst,
                                           config=config_tst,
                                           pixel_area=pixel_area_tst,
                                           int_label=None)

# Test function compute_width_over_one_section : check minimum width threshold
def test_compute_width_over_one_section_min_threshold(gser_buffers_large_gold, fpath_wm_base_large, buffer_length, dct_config_kwargs):
    """Test function compute_width_over_one_section : check minimum width threshold
    """

    # Set parameters for test
    pol_section_buffered_tst = gser_buffers_large_gold.loc[0]
    flt_buffer_area_tst = gser_buffers_large_gold.area.loc[0]

    watermask_tst = rio.open(fpath_wm_base_large, "r")
    pixel_area_tst = watermask_tst.transform[0] * np.abs(watermask_tst.transform[4])

    config_tst = ParamWidthComp(**dct_config_kwargs)
    config_tst.min_width = 800.
    config_tst.label_attr = ""

    # Test function
    width_tst, flg_bufful_tst, _, _, = compute_width_over_one_section(pol_section_buffered=pol_section_buffered_tst,
                                                                      flt_buffer_length=buffer_length,
                                                                      flt_buffer_area=flt_buffer_area_tst,
                                                                      watermask=watermask_tst,
                                                                      config=config_tst,
                                                                      pixel_area=pixel_area_tst,
                                                                      int_label=None)

    assert width_tst == 800.

# Test function compute_width_over_one_section : check returned width
@pytest.mark.parametrize("width_gold, bool_param1",
                         [(400., 0),
                          (np.nan, 1),
                          (np.nan, 2)])
def test_compute_width_over_one_section_right_width(dct_config_kwargs, gser_buffers_large_gold, fpath_wm_base_large, buffer_length, width_gold, bool_param1):
    """Test function compute_width_over_one_section : check returned width
    """

    # Set parameters for test
    pol_section_buffered_tst = gser_buffers_large_gold.loc[0]
    flt_buffer_area_tst = gser_buffers_large_gold.area.loc[0]

    watermask_tst = rio.open(fpath_wm_base_large, "r")
    pixel_area_tst = watermask_tst.transform[0] * np.abs(watermask_tst.transform[4])

    config_tst = ParamWidthComp(**dct_config_kwargs)
    config_tst.min_width = -1
    config_tst.label_attr = ""

    if bool_param1 == 1:
        pol_section_buffered_tst = Polygon()
    if bool_param1 == 2:
        pol_section_buffered_tst = None

    width_tst, flg_bufful_tst, _, _, = compute_width_over_one_section(pol_section_buffered=pol_section_buffered_tst,
                                                                      flt_buffer_length=buffer_length,
                                                                      flt_buffer_area=flt_buffer_area_tst,
                                                                      watermask=watermask_tst,
                                                                      config=config_tst,
                                                                      pixel_area=pixel_area_tst,
                                                                      int_label=None)

    if bool_param1 == 0:
        assert width_tst == width_gold
    else:
        assert  np.isnan(width_tst)

# Test function compute_width_over_one_section : check returned flag_buffer_full
def test_compute_width_over_one_section_right_flag_buffer_notfull(gser_buffers_large_gold, fpath_wm_base_large, buffer_length):
    """Test function compute_width_over_one_section : check returned flag_buffer_full
    """

    # Set parameters for test
    pol_section_buffered_tst = gser_buffers_large_gold.loc[0]
    flt_buffer_area_tst = gser_buffers_large_gold.area.loc[0]
    watermask_tst = rio.open(fpath_wm_base_large, "r")
    pixel_area_tst = watermask_tst.transform[0] * np.abs(watermask_tst.transform[4])
    config_tst = ParamWidthComp

    width_tst, flg_bufful_tst, _, _, = compute_width_over_one_section(pol_section_buffered=pol_section_buffered_tst,
                                                                      flt_buffer_length=buffer_length,
                                                                      flt_buffer_area=flt_buffer_area_tst,
                                                                      watermask=watermask_tst,
                                                                      config=config_tst,
                                                                      pixel_area=pixel_area_tst,
                                                                      int_label=None)

    assert flg_bufful_tst == 0

# Test function compute_width_over_one_section : check returned flag_buffer_full
def test_compute_width_over_one_section_right_flag_buffer_full(gser_buffers_short_gold, fpath_wm_base_large, buffer_length):
    """Test function compute_width_over_one_section : check returned flag_buffer_full
    """

    # Set parameters for test
    pol_section_buffered_tst = gser_buffers_short_gold.loc[0]
    flt_buffer_area_tst = gser_buffers_short_gold.area.loc[0]
    watermask_tst = rio.open(fpath_wm_base_large, "r")
    pixel_area_tst = watermask_tst.transform[0] * np.abs(watermask_tst.transform[4])
    config_tst = ParamWidthComp

    width_tst, flg_bufful_tst, _, _, = compute_width_over_one_section(pol_section_buffered=pol_section_buffered_tst,
                                                                      flt_buffer_length=buffer_length,
                                                                      flt_buffer_area=flt_buffer_area_tst,
                                                                      watermask=watermask_tst,
                                                                      config=config_tst,
                                                                      pixel_area=pixel_area_tst,
                                                                      int_label=None)

    assert flg_bufful_tst == 1

# Test function : quantify_intersection_ratio_between_buffer
def test_quantify_intersection_ratio_between_buffer(gdf_waterbuffer_gold):
    """Test function : quantify_intersection_ratio_between_buffer
    """

    # Reference data
    ser_beta_gold = pd.Series([4./12., 4./16.])

    # Test function
    ser_beta_tst = quantify_intersection_ratio_between_buffer(gdf_waterbuffer_gold)

    pd_testing.assert_series_equal(ser_beta_gold, ser_beta_tst)

# Test function : compute_widths_from_single_watermask_base : wrong inputs raise right exception
@pytest.mark.parametrize("bool_param_1, bool_param_2",
                         [(True, False),
                          (False, True)])
def test_compute_widths_from_single_watermask_base_wrong_inputs(gdf_sections_large_gold, fpath_wm_base_small, dct_config_kwargs, bool_param_1, bool_param_2):
    """Test function : compute_widths_from_single_watermask_base
    """

    # Set parameters for test
    sections_tst = gdf_sections_large_gold
    watermask_tst = rio.open(fpath_wm_base_small, "r")

    # Deactivate parameter to test
    if bool_param_1:
        watermask_tst = "a"
    if bool_param_2:
        sections_tst = "a"

    # Test
    with pytest.raises(TypeError):
        _, _ = compute_widths_from_single_watermask_base(watermask_tst,
                                                         sections_tst,
                                                         dct_config_kwargs)

# Test function : compute_widths_from_single_watermask_base : raise warning
def test_compute_widths_from_single_watermask_base_warning(caplog, gdf_sections_large_gold, fpath_wm_base_small):
    """Check if warning is raised
    """

    # Set parameters for test
    sections_tst = gdf_sections_large_gold
    sections_tst = sections_tst.to_crs(epsg=4326)
    watermask_tst = rio.open(fpath_wm_base_small, "r")

    # Test
    with caplog.at_level("WARNING"):
        _, _ = compute_widths_from_single_watermask_base(watermask_tst,
                                                         sections_tst)
    assert "Inputs in epsg:4326 are projected to epsg:3857, not effective away from equator." in caplog.text

# Test function compute_widths_from_single_watermask_base : save buffered sections
def test_compute_widths_from_single_watermask_base_save_buffered_sections_done(gdf_sections_large_gold, fpath_wm_base_small, fpath_buffers_large, dpath_outputs):
    """Test function compute_widths_from_single_watermask_base : save buffered sections
    """

    # Set parameters for test
    sections_tst = gdf_sections_large_gold
    watermask_tst = rio.open(fpath_wm_base_small, "r")
    fpath_base_out_tst = os.path.join(dpath_outputs, "buffered_sections.")

    _, _ = compute_widths_from_single_watermask_base(watermask_tst,
                                                     sections_tst,
                                                     export_buffered_sections=True,
                                                     fname_buffered_section=os.path.join(str(dpath_outputs), "buffered_sections.shp"))

    # Test if file exist
    for extension in ["shp", "cpg", "prj", "shx", "dbf"]:
        assert os.path.exists(fpath_base_out_tst + extension)

# Test function compute_widths_from_single_watermask_base : check if file content is correct
def test_compute_widths_from_single_watermask_base_save_buffered_sections_right(gdf_sections_large_gold, fpath_wm_base_small, fpath_buffers_large, dpath_outputs, buffer_length, gser_buffers_large_gold):
    """Test function compute_widths_from_single_watermask_base : save buffered sections
    """

    # Set parameters for test
    sections_tst = gdf_sections_large_gold
    watermask_tst = rio.open(fpath_wm_base_small, "r")
    fpath_out_tst = os.path.join(dpath_outputs, "buffered_sections.shp")

    _, _ = compute_widths_from_single_watermask_base(watermask_tst,
                                                     sections_tst,
                                                     export_buffered_sections=True,
                                                     fname_buffered_section=os.path.join(str(dpath_outputs), "buffered_sections.shp"),
                                                     buffer_length=buffer_length)
    gdf_buffer_tst = gpd.read_file(fpath_out_tst)
    assert_geoseries_equal(gdf_buffer_tst.geometry, gser_buffers_large_gold)


# Test function compute_widths_from_single_watermask_base : check label activation
@pytest.mark.parametrize("width_gold, label",
                         [(200., "label"),
                          (400., "")])
def test_compute_widths_from_single_watermask_base_label_activation(gdf_sections_large_gold, fpath_wm_label_large, width_gold, label, buffer_length):
    """Test function compute_widths_from_single_watermask_base : save buffered sections
    """

    # Set parameters for test
    sections_tst = gdf_sections_large_gold
    watermask_tst = rio.open(fpath_wm_label_large, "r")

    gdf_width_tst, _ = compute_widths_from_single_watermask_base(watermask_tst,
                                                                 sections_tst,
                                                                 label_attr=label,
                                                                 buffer_length=buffer_length)

    for index in range(4):
        assert gdf_width_tst.at[index, "width"] == width_gold

# Test function compute_widths_from_single_watermask_base : check dry section message
def test_compute_widths_from_single_watermask_base_dry_message(caplog, gdf_sections_large_gold, fpath_wm_dry_large, buffer_length):
    """Test function compute_widths_from_single_watermask_base : check dry section message
    """

    # Set parameters for test
    sections_tst = gdf_sections_large_gold
    watermask_tst = rio.open(fpath_wm_dry_large, "r")

    with caplog.at_level("INFO"):
        gdf_width_tst, _ = compute_widths_from_single_watermask_base(watermask_tst,
                                                                     sections_tst,
                                                                     index_attr="node_id",
                                                                     buffer_length=buffer_length,
                                                                     bool_print_dry=True)
        assert "Dry section: 0 (ID=10101)" in caplog.text

# Test function compute_widths_from_single_watermask_base : full outputs
def test_compute_widths_from_single_watermask_base_outputs(gdf_widths_gold, gdf_sections_large_gold, fpath_wm_base_large, buffer_length):
    """Test function compute_widths_from_single_watermask_base : full outputs
    """

    # Set parameters for test
    sections_tst = gdf_sections_large_gold
    watermask_tst = rio.open(fpath_wm_base_large, "r")

    gdf_width_tst, _ = compute_widths_from_single_watermask_base(watermask_tst,
                                                                 sections_tst,
                                                                 buffer_length=buffer_length)
    assert_geodataframe_equal(gdf_width_tst, gdf_widths_gold, check_dtype=False)







