import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
import pytest
import rasterio
from shapely.geometry import Point
from geopandas.testing import assert_geodataframe_equal

from mirrowrs.watermask import WaterMask, exclude_value_from_flattened_band
from mirrowrs.tools import FileExtensionError, DimensionError

str_fpath_wm_tif_test = "inputs/wm_tus.tif"
str_fpath_wm_small_tif_test = "inputs/wm_small_tus.tif"
str_fpath_wm_small_exclude_tif_test = "inputs/wm_small_tus_additionalvalue.tif"
str_fpath_wm_small_island_tif_test = "inputs/wm_small_tus_island.tif"
str_fpath_wm_polygon_full_shp_test = "inputs/wm_small_tus_islands_full.shp"
str_fpath_wm_polygon_shp_test = "inputs/wm_small_tus_islands.shp"

# Test instanciation
def test_init_basic():
    obj = WaterMask()
    assert isinstance(obj, WaterMask)

# Test instanciation : check if default values are set correctly
def test_init_default_values():
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
def test_from_tif_creates_instance():
    obj = WaterMask.from_tif(watermask_tif=str_fpath_wm_tif_test,
                                str_origin="my_source",
                                str_proj="proj")
    assert isinstance(obj, WaterMask)
    assert obj.str_provider == "my_source"
    assert obj.coordsyst == "proj"

# Test @classmethod : check attribute values
def test_init_attributes():
    obj = WaterMask.from_tif(watermask_tif=str_fpath_wm_tif_test,
                                str_origin="my_source",
                                str_proj="proj")

    assert obj.str_provider == "my_source"
    assert obj.str_fpath_infile == str_fpath_wm_tif_test
    assert obj.coordsyst == "proj"

    assert obj.bbox == (133000.0, 5419000.0, 134200.0, 5420000.0)
    assert obj.crs == "EPSG:2154"
    assert obj.crs_epsg == 2154
    assert obj.width == 120
    assert obj.height == 100
    assert obj.dtypes == rasterio.uint8
    assert obj.nodata == 255
    assert obj.res == 10.
    assert obj.dtype_label_out == rasterio.uint8

# Test @classmethod : check if wrong inputs raise right Exception : TypeError
def test_from_tif_origin_not_str():
    with pytest.raises(TypeError):
        WaterMask.from_tif(watermask_tif=str_fpath_wm_tif_test,
                           str_origin=123,
                           str_proj="proj")

# Test @classmethod : check if wrong inputs raise right Exception : FileExistError
def test_from_tif_wm_does_not_exist():
    with pytest.raises(FileExistsError):
        WaterMask.from_tif(watermask_tif="not_wm_tus.tif",
                           str_origin="my_source",
                           str_proj="proj")

# Test @classmethod : check if wrong inputs raise right Exception : FileExtensionError
def test_from_tif_wm_is_not_tif():
    with pytest.raises(FileExtensionError):
        WaterMask.from_tif(watermask_tif="inputs/wm_tus.shp",
                           str_origin="my_source",
                           str_proj="proj")

# Test @classmethod : check if wrong inputs raise right Exception : NotImplementedError
def test_from_tif_unexpected_projection():
    with pytest.raises(NotImplementedError):
        WaterMask.from_tif(watermask_tif=str_fpath_wm_tif_test,
                           str_origin="my_source",
                           str_proj="other")

# Test __str__ method in WaterMask class
def test_str_method():

    obj = WaterMask()
    assert str(obj) == "Empty WaterMask."

    obj = WaterMask.from_tif(watermask_tif=str_fpath_wm_tif_test,
                           str_origin="my_source",
                           str_proj="proj")
    assert str(obj) == "WaterMask product from my_source."

    obj = WaterMask.from_tif(watermask_tif=str_fpath_wm_tif_test,
                             str_proj="proj")

    assert str(obj) == "WaterMask product from inputs/wm_tus.tif."

# Test exclude_value_from_flattened_band function : check if wrong inputs raise right Exception
@pytest.mark.parametrize("band_in, value_to_exclude, expected_error",
                         [(1, 0., TypeError),
                          (np.ones((2,2)), 0., DimensionError),
                          (np.ones((4,)), "a", ValueError)])
def test_exclude_value_from_flattened_band_wrong_input_type(band_in, value_to_exclude, expected_error):
    with pytest.raises(expected_error):
        _ = exclude_value_from_flattened_band(npar_band_flat=band_in,
                                              value_to_exclude=value_to_exclude)

# Test exclude_value_from_flattened_band function
@pytest.mark.parametrize("excluded_value", [0., np.nan, np.inf])
def test_exclude_value_from_flattened_band(excluded_value):

    npar_input = np.ones((4,))
    npar_input[0] = excluded_value

    indices_test = exclude_value_from_flattened_band(npar_band_flat=npar_input,
                                                     value_to_exclude=excluded_value)
    indices_gold = np.array([1,2,3], dtype=np.int64)

    assert np.array_equal(indices_test, indices_gold)

# Test @staticmethod from band_to_pixc - exclude_values=None
def test_band_to_pixc_without_excluded_value():

    # Gold variables
    int_height = 10
    int_width = 12
    flt_min_lon = 133000
    flt_max_lat = 5420000
    flt_resolution = 10.
    int_epsg = 2154

    npar_int_band_gold = 255 * np.ones((int_height, int_width), dtype=np.uint8)
    npar_int_band_gold[:, 4:8] = 1

    npar_int_band_flat_gold = npar_int_band_gold.flatten()
    indices_gold = exclude_value_from_flattened_band(npar_band_flat=npar_int_band_flat_gold,
                                                     value_to_exclude=255)

    df_gold = pd.DataFrame(index=indices_gold, columns=["i", "j", "label", "clean"])
    df_gold["i"] = [ i//int_width for i in indices_gold]
    df_gold["j"] = [ i%int_width for i in indices_gold]
    df_gold["label"] = 1
    df_gold["clean"] = 1
    for col_name in ["i", "j"]:
        df_gold[col_name] = df_gold[col_name].astype(np.int64)
    for col_name in ["label", "clean"]:
        df_gold[col_name] = df_gold[col_name].astype(np.uint8)

    npar_float_x = np.arange(start = flt_min_lon + 0.5*flt_resolution,
                             stop=flt_min_lon + 0.5*flt_resolution + flt_resolution*int_width,
                             step=flt_resolution)
    npar_float_y = np.arange(start = flt_max_lat - 0.5*flt_resolution,
                             stop=flt_max_lat - 0.5*flt_resolution - flt_resolution*int_height,
                             step=-flt_resolution)
    gser_gold = gpd.GeoSeries(
        [ Point(npar_float_x[j], npar_float_y[i]) for (i,j) in zip(df_gold["i"], df_gold["j"])],
        index=df_gold.index,
        crs=CRS(int_epsg)
    )
    gdf_band_as_pixc_gold = gpd.GeoDataFrame(
        df_gold,
        geometry=gser_gold,
        crs=CRS(int_epsg)
    )

    with rasterio.open(str_fpath_wm_small_tif_test, "r") as raster_src:
        gdf_band_as_pixc_test = WaterMask.band_to_pixc(raster_src)

    assert_geodataframe_equal(gdf_band_as_pixc_test, gdf_band_as_pixc_gold)

# Test @staticmethod from band_to_pixc - exclude_values=2
def test_band_to_pixc_with_excluded_value():

    # Gold variables
    int_height = 10
    int_width = 12
    flt_min_lon = 133000
    flt_max_lat = 5420000
    flt_resolution = 10.
    int_epsg = 2154

    npar_int_band_gold = 255 * np.ones((int_height, int_width), dtype=np.uint8)
    npar_int_band_gold[:, 4:8] = 1
    npar_int_band_gold[0:2, 4:6] = 255

    npar_int_band_flat_gold = npar_int_band_gold.flatten()
    indices_gold = exclude_value_from_flattened_band(npar_band_flat=npar_int_band_flat_gold,
                                                     value_to_exclude=255)

    df_gold = pd.DataFrame(index=indices_gold, columns=["i", "j", "label", "clean"])
    df_gold["i"] = [ i//int_width for i in indices_gold]
    df_gold["j"] = [ i%int_width for i in indices_gold]
    df_gold["label"] = 1
    df_gold["clean"] = 1
    for col_name in ["i", "j"]:
        df_gold[col_name] = df_gold[col_name].astype(np.int64)
    for col_name in ["label", "clean"]:
        df_gold[col_name] = df_gold[col_name].astype(np.uint8)

    npar_float_x = np.arange(start = flt_min_lon + 0.5*flt_resolution,
                             stop=flt_min_lon + 0.5*flt_resolution + flt_resolution*int_width,
                             step=flt_resolution)
    npar_float_y = np.arange(start = flt_max_lat - 0.5*flt_resolution,
                             stop=flt_max_lat - 0.5*flt_resolution - flt_resolution*int_height,
                             step=-flt_resolution)
    gser_gold = gpd.GeoSeries(
        [ Point(npar_float_x[j], npar_float_y[i]) for (i,j) in zip(df_gold["i"], df_gold["j"])],
        index=df_gold.index,
        crs=CRS(int_epsg)
    )
    gdf_band_as_pixc_gold = gpd.GeoDataFrame(
        df_gold,
        geometry=gser_gold,
        crs=CRS(int_epsg)
    )

    with rasterio.open(str_fpath_wm_small_exclude_tif_test, "r") as raster_src:
        gdf_band_as_pixc_test = WaterMask.band_to_pixc(raster_src, exclude_values=2)

    assert_geodataframe_equal(gdf_band_as_pixc_test, gdf_band_as_pixc_gold)

# Test @staticmethod from band_to_pixc - check if wrong inputs raise right Exception - NotImplementedError
def test_band_to_pixc_iterable_excluded_value():
    with pytest.raises(NotImplementedError):
        with rasterio.open(str_fpath_wm_small_tif_test, "r") as raster_src:
            gdf_band_as_pixc_test = WaterMask.band_to_pixc(raster_src,
                                                           exclude_values=[0, 1])

# Test method update_clean_flag: check if wrong inputs raise right Exception
@pytest.mark.parametrize("mask_clean, expected_exception", [(0., TypeError),
                                                            ([4, "a"], TypeError),
                                                            ([0], ValueError)])
def test_update_clean_flag_wrong_inputs(mask_clean, expected_exception):

    wm_tst = WaterMask.from_tif(str_fpath_wm_small_tif_test)
    with pytest.raises(expected_exception):
       wm_tst.update_clean_flag(mask=mask_clean)

# Test method update_clean_flag
def test_update_clean_flag():

    # Test variable
    wm_tst = WaterMask.from_tif(str_fpath_wm_small_tif_test)
    wm_tst.update_clean_flag(mask=[4,5])

    # Gold variables
    int_height = 10
    int_width = 12
    flt_min_lon = 133000
    flt_max_lat = 5420000
    flt_resolution = 10.
    int_epsg = 2154

    npar_int_band_gold = 255 * np.ones((int_height, int_width), dtype=np.uint8)
    npar_int_band_gold[:, 4:8] = 1

    npar_int_band_flat_gold = npar_int_band_gold.flatten()
    indices_gold = exclude_value_from_flattened_band(npar_band_flat=npar_int_band_flat_gold,
                                                     value_to_exclude=255)

    df_gold = pd.DataFrame(index=indices_gold, columns=["i", "j", "label", "clean"])
    df_gold["i"] = [i // int_width for i in indices_gold]
    df_gold["j"] = [i % int_width for i in indices_gold]
    df_gold["label"] = 1
    df_gold["clean"] = 1
    df_gold.loc[[4,5], "clean"] = 0
    for col_name in ["i", "j"]:
        df_gold[col_name] = df_gold[col_name].astype(np.int64)
    for col_name in ["label", "clean"]:
        df_gold[col_name] = df_gold[col_name].astype(np.uint8)

    npar_float_x = np.arange(start=flt_min_lon + 0.5 * flt_resolution,
                             stop=flt_min_lon + 0.5 * flt_resolution + flt_resolution * int_width,
                             step=flt_resolution)
    npar_float_y = np.arange(start=flt_max_lat - 0.5 * flt_resolution,
                             stop=flt_max_lat - 0.5 * flt_resolution - flt_resolution * int_height,
                             step=-flt_resolution)
    gser_gold = gpd.GeoSeries(
        [Point(npar_float_x[j], npar_float_y[i]) for (i, j) in zip(df_gold["i"], df_gold["j"])],
        index=df_gold.index,
        crs=CRS(int_epsg)
    )
    gdf_band_as_pixc_gold = gpd.GeoDataFrame(
        df_gold,
        geometry=gser_gold,
        crs=CRS(int_epsg)
    )

    assert_geodataframe_equal(wm_tst.gdf_wm_as_pixc, gdf_band_as_pixc_gold)

# Test method update_label_flag: check if wrong inputs raise right Exception
@pytest.mark.parametrize("dct_label, expected_exception",
                         [({2: 0.}, TypeError),
                          ({2: [4, "a"]}, TypeError),
                          ({2: [0]}, ValueError),
                          ({"a": [4, 5]}, ValueError),
                          ("a", TypeError),
                          ({66000: [4,5]}, NotImplementedError)])
def test_update_label_flag_wrong_inputs(dct_label, expected_exception):

    wm_tst = WaterMask.from_tif(str_fpath_wm_small_tif_test)
    with pytest.raises(expected_exception):
       wm_tst.update_label_flag(dct_label=dct_label)

# Test method update_label_flag
@pytest.mark.parametrize("label, dtype_out, val_nodata",
                         [(2, rasterio.uint8, 255),
                          (500, rasterio.uint16, 65535)])
def test_update_label_flag(label, dtype_out, val_nodata):

    # Gold variables
    int_height = 10
    int_width = 12
    flt_min_lon = 133000
    flt_max_lat = 5420000
    flt_resolution = 10.
    int_epsg = 2154

    npar_int_band_gold = 255 * np.ones((int_height, int_width), dtype=np.uint8)
    npar_int_band_gold[:, 4:8] = 1

    npar_int_band_flat_gold = npar_int_band_gold.flatten()
    indices_gold = exclude_value_from_flattened_band(npar_band_flat=npar_int_band_flat_gold,
                                                     value_to_exclude=255)

    df_gold = pd.DataFrame(index=indices_gold, columns=["i", "j", "label", "clean"])
    df_gold["i"] = [i // int_width for i in indices_gold]
    df_gold["j"] = [i % int_width for i in indices_gold]
    df_gold["label"] = 1
    df_gold["clean"] = 1
    for col_name in ["i", "j"]:
        df_gold[col_name] = df_gold[col_name].astype(np.int64)
    for col_name in ["label", "clean"]:
        df_gold[col_name] = df_gold[col_name].astype(np.uint8)

    # Update label for test
    if label > 255:
        df_gold["label"] = df_gold["label"].astype(np.uint16)
        df_gold.loc[[4, 5, 6, 7], "label"] = label
    else:
        df_gold.loc[[4, 5, 6, 7], "label"] = label

    npar_float_x = np.arange(start=flt_min_lon + 0.5 * flt_resolution,
                             stop=flt_min_lon + 0.5 * flt_resolution + flt_resolution * int_width,
                             step=flt_resolution)
    npar_float_y = np.arange(start=flt_max_lat - 0.5 * flt_resolution,
                             stop=flt_max_lat - 0.5 * flt_resolution - flt_resolution * int_height,
                             step=-flt_resolution)
    gser_gold = gpd.GeoSeries(
        [Point(npar_float_x[j], npar_float_y[i]) for (i, j) in zip(df_gold["i"], df_gold["j"])],
        index=df_gold.index,
        crs=CRS(int_epsg)
    )
    gdf_band_as_pixc_gold = gpd.GeoDataFrame(
        df_gold,
        geometry=gser_gold,
        crs=CRS(int_epsg)
    )

    # Test variable
    wm_tst = WaterMask.from_tif(str_fpath_wm_small_tif_test)
    wm_tst.update_label_flag(dct_label={label: [4,5,6,7]})

    # Assert method
    assert_geodataframe_equal(wm_tst.gdf_wm_as_pixc, gdf_band_as_pixc_gold)
    assert wm_tst.dtype_label_out == dtype_out
    assert wm_tst.nodata == val_nodata

# Test method get_band
@pytest.mark.parametrize("bool_clean, bool_label, bool_masked_array",
                         [(True, True, False),
                          (True, False, False),
                          (False, True, False),
                          (False,False, False),
                          (True, True, True)])
def test_get_band(bool_clean, bool_label, bool_masked_array):

    # Gold variables
    int_height = 10
    int_width = 12
    npar_int_band_gold = 255 * np.ones((int_height, int_width), dtype=np.uint8)
    npar_int_band_gold[:,4:8] = 1

    # Add label
    if bool_label:
        npar_int_band_gold[:5, 4:8] = 2
        npar_int_band_gold[5:, 4:8] = 3
    if bool_clean:
        npar_int_band_gold[0,4:8] = 255
    if bool_masked_array:
        npar_int_band_gold = np.ma.array(
                npar_int_band_gold,
                mask=(npar_int_band_gold == 255),
            )

    # Test variable
    wm_tst = WaterMask.from_tif(str_fpath_wm_small_tif_test)
    wm_tst.update_clean_flag(mask=[4,5,6,7])
    wm_tst.update_label_flag(dct_label={2: [4, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55],
                                        3: [64,65,66,67,76,77,78,79,88,89,90,91,100,101,102,103,112,113,114,115]})

    npar_int_band_tst = wm_tst.get_band(bool_clean, bool_label, bool_masked_array)

    # Assert method
    if not bool_masked_array:
        np.array_equal(npar_int_band_tst, npar_int_band_gold)
    else:
        np.ma.allequal(npar_int_band_tst, npar_int_band_gold)

# Test method get_polygons
@pytest.mark.parametrize("bool_exterior_only, bool_indices",
                         [(True, False),
                          (False, False),
                          (True, True),
                          (True, False)])
def test_get_polygons(bool_exterior_only, bool_indices):

    # Gold variable
    int_height = 10
    int_width = 12
    npar_int_band_gold = 255 * np.ones((int_height, int_width), dtype=np.uint8)

    npar_int_band_gold[:, 4:8] = 1
    npar_int_band_gold[4:8, 5:7] = 255
    indices = list(np.where(npar_int_band_gold.flatten() != 255))

    if bool_exterior_only:
        str_fpath_wm_polygons_shp = str_fpath_wm_polygon_full_shp_test
    else:
        str_fpath_wm_polygons_shp = str_fpath_wm_polygon_shp_test

    if not bool_indices:
        indices = None

    gdf_tmp_gold = gpd.read_file(str_fpath_wm_polygons_shp)
    gser_gold = gdf_tmp_gold["geometry"]
    del gdf_tmp_gold

    gdf_wm_polygons_gold = gpd.GeoDataFrame(
        pd.DataFrame({"label": [1],
                      "clean": [1],
                      "indices": None}),
        geometry=gser_gold,
        crs=gser_gold.crs
    )
    gdf_wm_polygons_gold["label"] = gdf_wm_polygons_gold["label"].astype(np.uint8)
    gdf_wm_polygons_gold["clean"] = gdf_wm_polygons_gold["clean"].astype(np.uint8)
    gdf_wm_polygons_gold["indices"] = indices

    # Test variable
    wm_tst = WaterMask.from_tif(str_fpath_wm_small_island_tif_test)
    gdf_wm_polygons_tst = wm_tst.get_polygons(bool_clean=False,
                                              bool_label=False,
                                              bool_exterior_only=bool_exterior_only,
                                              bool_indices=bool_indices)

    assert_geodataframe_equal(gdf_wm_polygons_tst, gdf_wm_polygons_gold)





