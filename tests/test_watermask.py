import pytest
import rasterio

from mirrowrs.watermask import WaterMask
from mirrowrs.tools import FileExtensionError

str_fpath_wm_tif_test = "inputs/wm_tus.tif"

# Test instanciation
def test_init_basic():
    obj = WaterMask()
    assert isinstance(obj, WaterMask)

# Test default value
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
    print(obj.bbox)
    assert isinstance(obj, WaterMask)
    assert obj.str_provider == "my_source"
    assert obj.coordsyst == "proj"

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


# Test @classmethod : check if wrong inputs raise right Exception
def test_from_tif_origin_not_str():
    with pytest.raises(TypeError):
        WaterMask.from_tif(watermask_tif=str_fpath_wm_tif_test,
                           str_origin=123,
                           str_proj="proj")

def test_from_tif_wm_does_not_exist():
    with pytest.raises(FileExistsError):
        WaterMask.from_tif(watermask_tif="not_wm_tus.tif",
                           str_origin="my_source",
                           str_proj="proj")

def test_from_tif_wm_is_not_tif():
    with pytest.raises(FileExtensionError):
        WaterMask.from_tif(watermask_tif="inputs/wm_tus.shp",
                           str_origin="my_source",
                           str_proj="proj")

def test_from_tif_unexpected_projection():
    with pytest.raises(NotImplementedError):
        WaterMask.from_tif(watermask_tif=str_fpath_wm_tif_test,
                           str_origin="my_source",
                           str_proj="other")

# Test unitaire de __str__
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


