# Copyright (C) 2024-2025 CS GROUP, https://csgroup.eu
# Copyright (C) 2024 CNES.
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

# More complete example use of MIRROWRS tools:
# - How to add a new/specific type of watermask via child classes WaterMaskCE and BASProcessorCE
# - How to compute a width product as a combination of BAS-based widths products (from scenario 0 and 11)
# - A first uncertainty model for node-scale width

"""
A complete example of MIRROWRS use
"""

import logging
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio

from mirrowrs.basprocessor import BASProcessor
from mirrowrs.rivergeomproduct import RiverGeomProduct
from mirrowrs.tools import FileExtensionError
from mirrowrs.watermask import WaterMask

_logger = logging.getLogger("mirrowrs_on_surfwater")

# Config BAS
DCT_CONFIG_O = {
    "clean": {
        "bool_clean": True,
        "type_clean": "base",
        "fpath_wrkdir": ".",
        "gdf_waterbodies": None,
    },
    "label": {"bool_label": True, "type_label": "base", "fpath_wrkdir": "."},
    "reduce": {
        "how": "hydrogeom",
        "attr_nb_chan_max": "n_chan_max",
        "attr_locxs": "loc_xs",
        "attr_nodepx": "x_proj",
        "attr_nodepy": "y_proj",
        "attr_tolerance_dist": "tol_dist",
        "attr_meander_length": "meand_len",
        "attr_sinuosity": "sinuosity",
        "flt_tol_len": 0.05,
        "flt_tol_dist": "tol_dist",
    },
    "widths": {"scenario": 11},
}


def compute_nodescale_width(gdf_widths_ortho=None, gdf_widths_chck=None):
    """Compute node-scale widths

    :param gdf_widths_ortho:
    :param gdf_widths_chck:
    :param method:
    :return:
    """

    _logger = logging.getLogger("BAS PROCESSING")
    _logger.info("Get width_1-related info : width_1 + beta")
    gdf_widths_out = gdf_widths_ortho.copy(deep=True)
    # Width from ortho sections :: width_1
    gdf_widths_out.rename(mapper={"width": "width_1"}, axis=1, inplace=True)

    _logger.info("Get width_2-related info : width_2 + theta")
    # Width from parallel sections : width_2
    gdf_widths_out.insert(
        loc=len(gdf_widths_out.columns) - 1,
        column="width_2",
        value=gdf_widths_chck["width"],
    )

    # Add theta and sin_theta columns to output width dataframe
    # Theta is an angle in degree
    _logger.info("Set theta components")
    gdf_widths_out.insert(
        loc=len(gdf_widths_out.columns) - 1,
        column="theta",
        value=gdf_widths_chck["theta"],
    )
    # sin(theta) was previously computed from angle in radians
    gdf_widths_out.insert(
        loc=len(gdf_widths_out.columns) - 1,
        column="sin_theta",
        value=gdf_widths_chck["sin_theta"],
    )

    # Need to convert theta back to radians to apply np.cos
    gdf_widths_out.insert(
        loc=len(gdf_widths_out.columns) - 1, column="cos_theta", value=np.nan
    )
    gdf_widths_out["cos_theta"] = gdf_widths_chck["theta"].apply(
        lambda theta_deg: np.cos(theta_deg * np.pi / 180.0)
    )

    # Add final width product column
    _logger.info("Set width")
    gdf_widths_out.insert(
        loc=len(gdf_widths_out.columns) - 1, column="width", value=np.nan
    )

    # beta = 0 OR W2 == nan : W = W1
    # If no intersection between base section and the neighbouring one, check section is irrelevant
    # => beta=0 + no contribution from W2
    ser_mask_beta = gdf_widths_out["beta"].apply(
        lambda beta: True if beta == 0 else False
    )
    ser_mask_w2 = gdf_widths_out["width_2"].apply(
        lambda w2: True if np.isnan(w2) else False
    )
    ser_mask = ser_mask_beta | ser_mask_w2
    gdf_widths_out.loc[ser_mask, "width"] = gdf_widths_out.loc[ser_mask, "width_1"]

    # Else beta>0 AND W2!=nan : W = (W1 + |cos(theta)|xW2)/(1+|cos(theta)|)
    ser_cos_theta_tmp = gdf_widths_out["cos_theta"].apply(np.abs)
    gdf_widths_out.loc[~ser_mask, "width"] = gdf_widths_out.loc[
        ~ser_mask, "width_1"
    ] + gdf_widths_out.loc[~ser_mask, "width_2"].mul(ser_cos_theta_tmp)
    gdf_widths_out.loc[~ser_mask, "width"] = gdf_widths_out.loc[~ser_mask, "width"].div(
        ser_cos_theta_tmp[~ser_mask].apply(lambda ct: ct + 1.0)
    )

    del ser_cos_theta_tmp

    return gdf_widths_out


def compute_nodescale_widtherror(gdf_widths, flt_watermask_resol):
    """Compute node-scale uncertainty

    :param gdf_widths:
    :param flt_watermask_resol:
    :param method:
    :return:
    """

    # Observation/measurement error
    # related to the edges of the watermask
    ser_sigo = gdf_widths["nb_banks"].mul(flt_watermask_resol / np.sqrt(12.0))

    # Representativeness error
    ser_sigr = pd.Series(
        flt_watermask_resol / np.sqrt(12.0) * np.ones((len(gdf_widths),)),
        index=gdf_widths.index,
    )

    # Structure error : simplified version
    ser_sigs = pd.Series(
        flt_watermask_resol / np.sqrt(12.0) * np.ones((len(gdf_widths),)),
        index=gdf_widths.index,
    )

    ser_errtot = np.sqrt(ser_sigo**2.0 + ser_sigr**2.0 + ser_sigs**2.0)

    return ser_errtot, ser_sigo, ser_sigr, ser_sigs


class WaterMaskCHM(WaterMask):

    def __init__(self):

        # Init parent class
        super().__init__()

    @classmethod
    def from_surfwater(cls, surfwater_tif=None):
        """Instanciate from a Surfwater watermask

        :param surfwater_tif: str
            Full path to surfwater mask stored in a GeoTiff file
        :return: WaterMaskCHM
        """

        # Instanciate object
        klass = WaterMaskCHM()
        klass.str_provider = "SurfWater"
        klass.str_fpath_infile = surfwater_tif

        # Set watermask rasterfile
        if not os.path.isfile(surfwater_tif):
            _logger.error("Watermask.from_surfwater: Input tif file does not exist..")
            raise FileExistsError("Input tif file does not exist..")

        else:
            if not surfwater_tif.endswith(".tif"):
                raise FileExtensionError

        # Set raster coordinate system
        klass.coordsyst = "proj"

        with rio.open(surfwater_tif, "r") as src:

            klass.crs = src.crs
            klass.crs_epsg = src.crs.to_epsg()
            klass.bbox = (
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            )

            klass.transform = src.transform
            klass.res = src.transform.a
            klass.width = src.width
            klass.height = src.height
            klass.nodata = src.nodata

            klass.dtypes = src.dtypes[0]
            klass.dtype_label_out = src.dtypes[0]

            klass.gdf_wm_as_pixc = klass.band_to_pixc(src, exclude_values=0)

        return klass


class BASProcessorCHM(BASProcessor):

    def __init__(
        self,
        str_watermask_tif=None,
        gdf_sections=None,
        gdf_reaches=None,
        attr_reachid=None,
        str_proj="proj",
        str_provider=None,
        str_datetime=None,
    ):
        """Class constructor

        :param str_watermask_tif: str
            Full path to watermask
        :param gdf_sections: gpd.GeoDataFrame
            Set of cross-sections over which extract widths
        :param gdf_reaches: gpd.GeoDataFrame
            Set of river centerlines
        :param attr_reachid: str
            In gdf_reaches and gdf_sections: attribute name for reach_id
        :param str_proj: str
            ["proj", "lonlat"] : indicate if input watermask is in a geographic or geometric system
        :param str_provider: str
            Indicates the origin of watermask
        :param str_datetime: str
            Watermask scene datetime with format '%Y%m%dT%H%M%S'
        """

        # Init parent class
        super().__init__(
            str_watermask_tif,
            gdf_sections,
            gdf_reaches,
            attr_reachid,
            str_proj,
            str_provider,
            str_datetime,
        )

    def preprocessing(self, bool_load_wm=True, crs_proj_wm=None):
        """Preprocessing: load watermask, reproject sections et check bounding boxes intersections"""

        _logger.info("----- BASProcessorCHM = Preprocessing -----")

        # Load WaterMask object
        if bool_load_wm:
            if self.provider == "SW":  # Watermask is provided by Surfwater
                _logger.info("Load watermask from Surfwater")
                self.watermask = WaterMaskCHM.from_surfwater(self.f_watermask_in)

                # Check boundingbox compatibility
                self.check_bbox_compatibility()
            else:
                _logger.error(f"Provider {self.provider} not recognized")
                raise NotImplementedError
            _logger.info("Watermask loaded..")
        else:
            self.watermask = WaterMaskCHM()
            _logger.info("User chose not to load watermask..")

        # Reproject sections to watermask coordinate system
        _logger.info("Reproject sections to watermask coordinate system")
        try:
            self.gdf_reaches = self.gdf_reaches.to_crs(self.watermask.crs_epsg)
            self.gdf_sections = self.gdf_sections.to_crs(self.watermask.crs_epsg)
        except:
            if crs_proj_wm is not None:
                self.gdf_reaches = self.gdf_reaches.to_crs(crs_proj_wm)
                self.gdf_sections = self.gdf_sections.to_crs(crs_proj_wm)
            else:
                raise ValueError(
                    "Missing required 'crs_proj_wm' input to reproject reaches/sections without loading wm beforehand."
                )
        _logger.info("Reproject sections to watermask coordinate system done ..")

        print("")
        print("----- Preprocessing : Done -----")


class WidthProcessor:

    def __init__(
        self,
        str_watermask_tif=None,
        str_datetime=None,
        str_reaches_shp=None,
        str_nodes_shp=None,
    ):
        """Class constructor

        :param str_watermask_tif: str
            Full path to watermask from which compute river width
        :param str_datetime: str
            Watermask scene datetime with format '%Y%m%dT%H%M%S'
        :param str_reaches_shp: str
            Full path to shapefile with river centerlines geometries
        :param str_nodes_shp: str
            Full path to shapefile with river nodes at which compute cross-sections
        """

        _logger.info("Instanciate WidthProcessor")

        # Check inputs
        if str_watermask_tif is None:
            raise ValueError("Missing watermask GeoTiff input file")
        if not os.path.isfile(str_watermask_tif):
            raise FileExistsError("Input watermask GeoTiff does not exist")
        if str_datetime is None:
            raise ValueError("Missing scene datetime information input")
        try:
            _ = datetime.strptime(str_datetime, "%Y%m%dT%H%M%S")
        except ValueError:
            raise ValueError(
                "input datetime {} does not match format '%Y%m%dT%H%M%S'.".format(str_datetime))
        if str_reaches_shp is None:
            raise ValueError("Missing reaches shapefile input")
        if not os.path.join(str_reaches_shp):
            raise FileExistsError("Input reaches shapefile does not exist")
        if str_nodes_shp is None:
            raise ValueError("Missing nodes shapefile input")
        if not os.path.join(str_nodes_shp):
            raise FileExistsError("Input nodes shapefile does not exist")
        _logger.info("Input checked")

        # Set attributes from inputs
        self.f_watermask_in = str_watermask_tif
        self.scene_name = os.path.basename(self.f_watermask_in).split(".")[0]
        self.scene_datetime = str_datetime
        self.reaches_shp = str_reaches_shp
        self.nodes_shp = str_nodes_shp
        _logger.info("Attributes from inputs set")

        # Derive other attributes
        self.gdf_reaches = gpd.read_file(self.reaches_shp)
        self.gdf_nodes = gpd.read_file(self.nodes_shp)
        _logger.info("Attributes derived from inputs set")

        # Initiate future computed attributes
        self.gdf_sections_ortho = None
        self.gdf_sections_chck = None

        self.bas_processor_o = None
        self.bas_processor_c = None

        self.gdf_nodescale_widths = None
        _logger.info("Future computed attributes initiated")

    def preprocessing(self, flt_factor_width=10.0):
        """Prepare tools to compute width ie: reaches and cross-section geometries in right system

        :param flt_factor_width: float
            Base width multiplying factor to draw section
        """

        _logger = logging.getLogger("WidthProcessing.preprocessing")

        # Get coordinate system from watermask
        with rio.open(self.f_watermask_in) as src:
            crs_wm_in = src.crs

        # Instanciate RiverGeom object
        _logger.info("Instanciate RiverGeomProduct object ..")
        try:
            dct_geom_attr = {
                "reaches": {"reaches_id": "reach_id"},
                "nodes": {
                    "reaches_id": "reach_id",
                    "nodes_id": "node_id",
                    "pwidth": "p_width",
                    "pwse": "p_wse",
                },
            }
            obj_rivergeom = RiverGeomProduct.from_shp(
                reaches_shp=self.reaches_shp,
                nodes_shp=self.nodes_shp,
                bool_edge=False,
                dct_attr=dct_geom_attr,
                crs_in=crs_wm_in,
            )
            _logger.info("Instanciation done..")
        except Exception as err:
            _logger.error(err)
            _logger.error("Instanciate RiverGeomProduct object KO ..")
            raise Exception

        # Set centerlines for section definition
        _logger.info("Set centerlines for section definition")
        try:
            obj_rivergeom.draw_allreaches_centerline()
        except Exception as err:
            _logger.error(err)
            _logger.error("Set centerlines for section definition KO ..")
            raise Exception

        # Traditionnal orthogonal sections
        _logger.info("Traditionnal orthogonal sections ..")
        try:
            self.gdf_sections_ortho = obj_rivergeom.draw_allreaches_sections(
                type="ortho", flt_factor_width=flt_factor_width
            )
        except Exception as err:
            _logger.error(err)
            _logger.error("Draw orthogonal sections KO ..")
            raise Exception

        # Checking parallel sections
        _logger.info("Parallel sections given the main direction of the reach")
        try:
            self.gdf_sections_chck = obj_rivergeom.draw_allreaches_sections(
                type="chck", flt_factor_width=flt_factor_width
            )
        except Exception as err:
            _logger.error(err)
            _logger.error("Draw parallel sections KO ..")
            raise Exception

        # Instanciate BASProcessorCHM objects
        try:
            _logger.info("Instanciate BASProcessor object for sections_ortho")
            self.bas_processor_o = BASProcessorCHM(
                str_watermask_tif=self.f_watermask_in,
                gdf_sections=self.gdf_sections_ortho,
                gdf_reaches=self.gdf_reaches,
                attr_reachid="reach_id",
                str_proj="proj",
                str_provider="SW",
            )
            self.bas_processor_o.preprocessing()
        except Exception as err:
            _logger.error(err)
            _logger.error("Instanciate width processor ortho KO ..")
            raise Exception

        try:
            _logger.info("Instanciate BASProcessor object for sections_chck")
            self.bas_processor_c = BASProcessorCHM(
                str_watermask_tif=self.f_watermask_in,
                gdf_sections=self.gdf_sections_chck,
                gdf_reaches=self.gdf_reaches,
                attr_reachid="reach_id",
                str_proj="proj",
                str_provider="SW",
            )
            self.bas_processor_c.preprocessing(
                bool_load_wm=False, crs_proj_wm=self.bas_processor_o.watermask.crs_epsg
            )  # To avoid loading watermask twice
        except Exception as err:
            _logger.error(err)
            _logger.error("Instanciate width processor check KO ..")
            raise Exception

    def basprocessing_ortho(
        self, out_dir=".", str_pekel_shp=None, str_type_clean=None, str_type_label=None
    ):
        """Perform BASProcessor processing (=clean+label) - only on BASProcessor associated to orthogonal sections

        :param out_dir: str
            Full path to directory where store outputs
        :param str_pekel_shp: str
            Full path to a vectorised version of a Pekel waterbody tile
        :param str_type_clean: str
            "base"/"waterbodies" : watermask cleaning method
        :param str_type_label: str
            "base" : watermask segmentation/labelling method
        """
        _logger = logging.getLogger("MIRROWRS::ORTHOGONAL-PROCESSING")
        _logger.info("Processing based on orthogonal sections")
        try:
            dct_cfg_o = DCT_CONFIG_O

            # Set cleaning and labelling method
            _logger.info(f"Set cleaning method to : {str_type_clean}")
            if str_type_clean is not None:
                if str_type_clean not in ["base", "waterbodies"]:
                    _logger.info(
                        "Undefined cleaning type .. Ignore .. use default value"
                    )
                else:
                    dct_cfg_o["clean"]["type_clean"] = str_type_clean

            _logger.info(f"Set labelling method to : {str_type_label}")
            if str_type_label is not None:
                if str_type_label not in ["base"]:
                    _logger.info(
                        "Undefined labelling type .. Ignore .. use default value"
                    )
                else:
                    dct_cfg_o["label"]["type_label"] = str_type_label

            # Set waterbody mask and output directory
            if str_type_clean == "waterbodies":
                gdf_waterbodies = gpd.read_file(str_pekel_shp)
                dct_cfg_o["clean"]["gdf_waterbodies"] = gdf_waterbodies
            dct_cfg_o["clean"]["fpath_wrkdir"] = out_dir
            dct_cfg_o["label"]["fpath_wrkdir"] = out_dir

            self.bas_processor_o.processing(dct_cfg_o)

        except Exception as err:
            _logger.error(err)
            _logger.error("Processing (clean+label) based on orthogonal section KO ..")
            raise Exception

    def basproccessing_ortho_widths(self, out_dir="."):
        """Width computation based on orthogonal sections

        :param out_dir: str
            Full path to directory where store outputs
        :return gdf_widths_ortho: gpd.GeoDataFrame
            Set of widths computed at orthogonal cross-sections
        :return str_fpath_wm_out: str
            Full path to prepared watermask
        """

        _logger.info("Width computation based on orthogonal sections")

        try:
            dct_cfg_o = DCT_CONFIG_O

            gser_proj_nodes = self.gdf_nodes["geometry"].to_crs(
                self.bas_processor_o.watermask.crs
            )
            _logger.info("reproject nodes to watermask crs done..")

            # Add required attributes for sections reduction
            attr_nodepx = dct_cfg_o["reduce"]["attr_nodepx"]
            self.bas_processor_o.gdf_sections.insert(
                loc=3, column=attr_nodepx, value=0.0
            )
            self.bas_processor_o.gdf_sections[attr_nodepx] = gser_proj_nodes.loc[
                self.bas_processor_o.gdf_sections.index
            ].x
            _logger.info("Add required attributes for sections reduction: x, done..")

            attr_nodepy = dct_cfg_o["reduce"]["attr_nodepy"]
            self.bas_processor_o.gdf_sections.insert(
                loc=4, column=attr_nodepy, value=0.0
            )
            self.bas_processor_o.gdf_sections[attr_nodepy] = gser_proj_nodes.loc[
                self.bas_processor_o.gdf_sections.index
            ].y
            _logger.info("Add required attributes for sections reduction: y, done..")

            attr_n_chan_max = dct_cfg_o["reduce"]["attr_nb_chan_max"]
            self.bas_processor_o.gdf_sections.insert(
                loc=5, column=attr_n_chan_max, value=0
            )
            self.bas_processor_o.gdf_sections[attr_n_chan_max] = self.gdf_nodes.loc[
                self.bas_processor_o.gdf_sections.index, attr_n_chan_max
            ]
            _logger.info("Add required attributes for sections reduction: nb_chan_max, done..")

            attr_tolerance_dist = dct_cfg_o["reduce"]["attr_tolerance_dist"]
            attr_meandr_len = dct_cfg_o["reduce"]["attr_meander_length"]
            attr_sinuosity = dct_cfg_o["reduce"]["attr_sinuosity"]
            self.bas_processor_o.gdf_sections.insert(
                loc=6, column=attr_tolerance_dist, value=0
            )

            self.bas_processor_o.gdf_sections[attr_tolerance_dist] = (
                0.5
                * self.gdf_nodes.loc[self.gdf_nodes.index, attr_meandr_len]
                / self.gdf_nodes.loc[self.gdf_nodes.index, attr_sinuosity]
            )
            _logger.info("Add required attributes for sections reduction: tolerance_dist, done..")

            _logger.info("Run bas_processor_o.postprocessing")
            gdf_widths_ortho, str_fpath_wm_out = self.bas_processor_o.postprocessing(
                dct_cfg=dct_cfg_o, str_fpath_dir_out=out_dir
            )
            _logger.info("bas_processor_o.postprocessing done..")

            return gdf_widths_ortho, str_fpath_wm_out

        except Exception as err:
            _logger.error(err)
            _logger.error("Processing based on orthogonal section KO ..")
        raise Exception

    def basprocessing_chck(self):
        """Prepare tools for processing based on parallel sections"""

        try:

            self.bas_processor_c.watermask = self.bas_processor_o.watermask

        except Exception as err:
            _logger.error(err)
            _logger.error("Processing based on parallel section : preparation KO ..")
            raise Exception

    def basprocessing_chck_widths(self, str_fpath_wm_in=None):
        """Width computation based on check sections

        :param str_fpath_wm_in: str
            Full path to prepared watermask
        :return gdf_widths_chck: gpd.GeoDataFrame
            Set of widths computed at parallel cross-sections
        :return str_fpath_wm_tif:str
            Full path to prepared watermask
        """

        _logger.info("Width computation based on check sections")
        try:
            dct_cfg_c = DCT_CONFIG_O
            dct_cfg_c["widths"]["scenario"] = 0

            gser_proj_nodes = self.gdf_nodes["geometry"].to_crs(
                self.bas_processor_o.watermask.crs_epsg
            )

            # Add required attributes for sections reduction
            attr_nodepx = dct_cfg_c["reduce"]["attr_nodepx"]
            self.bas_processor_c.gdf_sections.insert(
                loc=3, column=attr_nodepx, value=0.0
            )
            self.bas_processor_c.gdf_sections[attr_nodepx] = gser_proj_nodes.loc[
                self.bas_processor_c.gdf_sections.index
            ].x

            attr_nodepy = dct_cfg_c["reduce"]["attr_nodepy"]
            self.bas_processor_c.gdf_sections.insert(
                loc=4, column=attr_nodepy, value=0.0
            )
            self.bas_processor_c.gdf_sections[attr_nodepy] = gser_proj_nodes.loc[
                self.bas_processor_c.gdf_sections.index
            ].y

            attr_n_chan_max = dct_cfg_c["reduce"]["attr_nb_chan_max"]
            self.bas_processor_c.gdf_sections.insert(
                loc=5, column=attr_n_chan_max, value=0
            )
            self.bas_processor_c.gdf_sections[attr_n_chan_max] = self.gdf_nodes.loc[
                self.bas_processor_c.gdf_sections.index, attr_n_chan_max
            ]

            attr_tolerance_dist = dct_cfg_c["reduce"]["attr_tolerance_dist"]
            attr_meandr_len = dct_cfg_c["reduce"]["attr_meander_length"]
            attr_sinuosity = dct_cfg_c["reduce"]["attr_sinuosity"]
            self.bas_processor_c.gdf_sections.insert(
                loc=6, column=attr_tolerance_dist, value=0
            )
            self.bas_processor_c.gdf_sections[attr_tolerance_dist] = (
                0.5
                * self.gdf_nodes.loc[self.gdf_nodes.index, attr_meandr_len]
                / self.gdf_nodes.loc[self.gdf_nodes.index, attr_sinuosity]
            )

            gdf_widths_chck, str_fpath_wm_tif = self.bas_processor_c.postprocessing(
                dct_cfg=dct_cfg_c, str_fpath_wm_in=str_fpath_wm_in
            )

            return gdf_widths_chck, str_fpath_wm_tif

        except Exception as err:
            _logger.error(err)
            _logger.error("Processing based on check section KO ..")
        raise Exception

    def processing(
        self, out_dir=".", str_pekel_shp=None, str_type_clean=None, str_type_label=None
    ):
        """Produce river width derived from watermask

        :param out_dir: str
            Full path to directory where store outputs
        :param str_pekel_shp: str
            Full path to a vectorised version of a Pekel waterbody tile
        ::param str_type_clean: str
            "base"/"waterbodies" : watermask cleaning method
        :param str_type_label: str
            "base" : watermask segmentation/labelling method
        :return str_fpath_wm_tif: str
        :return str_fpath_wm_out: str
        """

        self.basprocessing_ortho(
            out_dir=out_dir,
            str_pekel_shp=str_pekel_shp,
            str_type_clean=str_type_clean,
            str_type_label=str_type_label,
        )
        gdf_widths_ortho, str_fpath_wm_out = self.basproccessing_ortho_widths(
            out_dir=out_dir
        )

        self.basprocessing_chck()
        gdf_widths_chck, str_fpath_wm_tif = self.basprocessing_chck_widths(
            str_fpath_wm_in=str_fpath_wm_out
        )

        # Compute node-scale widths
        _logger.info("Compute widths at node scale")
        try:
            self.gdf_nodescale_widths = compute_nodescale_width(
                gdf_widths_ortho, gdf_widths_chck
            )

            self.gdf_nodescale_widths.insert(loc=2, column="provider", value="SW")
            self.gdf_nodescale_widths.insert(loc=3, column="bool_ko", value=0)
            self.gdf_nodescale_widths["bool_ko"] = self.gdf_nodescale_widths[
                "width"
            ].apply(lambda w: np.logical_or(np.isnan(w), w == 0))

            _logger.info("Node-scale widths computed..")
        except Exception as err:
            _logger.error(err)
            _logger.info("Node-scale widths computed ko..")
            raise Exception

        # Compute node-scale width errors
        _logger.info("Compute width error at node scale")
        try:
            ser_errtot, ser_sigo, ser_sigr, ser_sigs = compute_nodescale_widtherror(
                self.gdf_nodescale_widths, self.bas_processor_o.watermask.res
            )
            self.gdf_nodescale_widths.insert(loc=3, column="width_u", value=ser_errtot)
            self.gdf_nodescale_widths.insert(loc=4, column="sigo", value=ser_sigo)
            self.gdf_nodescale_widths.insert(loc=5, column="sigr", value=ser_sigr)
            self.gdf_nodescale_widths.insert(loc=6, column="sigs", value=ser_sigs)

            _logger.info("Node-scale width errors computed..")
        except Exception as err:
            _logger.error(err)
            _logger.info("Node-scale width errors computed ko..")
            raise Exception

        return str_fpath_wm_tif, str_fpath_wm_out

    def postprocessing(self, output_dir=".", more_outputs=False):
        """Save processed riverwidths into files

        :param output_dir: str
            Full path to directory where store outputs
        :param more_outputs: boolean
            If True, store all detailed outputs
        """

        _logger = logging.getLogger("WidthProcessing.postprocessing")

        _logger.info("Save node-scale width as csv ")
        df_nodes_width = self.gdf_nodescale_widths.loc[
            :, ["reach_id", "node_id", "provider", "width", "width_u"]
        ].copy(deep=True)
        df_nodes_width = df_nodes_width.dropna()
        df_nodes_width["datetime"] = self.scene_datetime
        width_nodes_csv = self.scene_name + "_nodescale_BAS_widths.csv"
        df_nodes_width.to_csv(os.path.join(output_dir, width_nodes_csv))
        _logger.info("Node-scale width saved to csv..")

        # Save cross-sections with associated width as shp in epsg:4326
        if more_outputs:
            _logger.info(
                "Save cross-sections (node-scale) with associated width as shp in epsg:4326"
            )
            width_nodes_shp = self.scene_name + "_nodescale_BAS_widths.shp"
            self.gdf_nodescale_widths.insert(
                loc=len(self.gdf_nodescale_widths.columns) - 1,
                column="datetime",
                value=self.scene_datetime,
            )
            self.gdf_nodescale_widths["reach_id"] = self.gdf_nodescale_widths[
                "reach_id"
            ].astype(str)
            self.gdf_nodescale_widths["node_id"] = (
                self.gdf_nodescale_widths["node_id"].astype(int).astype(str)
            )
            self.gdf_nodescale_widths.to_file(os.path.join(output_dir, width_nodes_shp))
            _logger.info("Node-scale width saved to shp..")


def parse_inputs():
    """Read script inputs
    :return args: ArgumentParser
    """

    parser = ArgumentParser(
        description="Extract river width at SWORD nodes and reaches from a watermask"
    )

    parser.add_argument(
        "-w",
        "--watermask_tif",
        type=str,
        default=None,
        help="Full path of the input watermask as GeoTiff to process",
    )
    parser.add_argument(
        "-dt",
        "--scn_datetime",
        type=str,
        default="20200111T105853",
        help="Scene datetime",
    )
    parser.add_argument(
        "-r",
        "--reaches_shp",
        type=str,
        default=None,
        help="Full path to shp with reach-line geometries",
    )
    parser.add_argument(
        "-n",
        "--nodes_shp",
        type=str,
        default=None,
        help="Full path to shp with node-point geometries",
    )
    parser.add_argument(
        "-tc",
        "--type_clean",
        type=str,
        default="base",
        help="Watermask cleaning procedure",
    )
    parser.add_argument(
        "-tl",
        "--type_label",
        type=str,
        default="base",
        help="Watermask labelling procedure",
    )
    parser.add_argument(
        "-fw",
        "--factor_width",
        type=float,
        default=10.0,
        help="Multiplying factor applied to PRD width to draw section",
    )
    parser.add_argument(
        "-o",
        "--outputdir",
        type=str,
        default=".",
        help="Full path to directory where to store logfiles and outputs.",
    )
    parser.add_argument(
        "-l",
        "--loglevel",
        type=str,
        default="info",
        help="Logging level: debug, info, warning, error",
    )
    parser.add_argument(
        "-lf",
        "--logfile",
        action="store_true",
        help="Logging file saved",
    )
    parser.add_argument(
        "-mo",
        "--more_outputs",
        action="store_true",
        help="If activated, shapefile and cleaned watermasks will be saved too",
    )

    # Set input arguments
    args = parser.parse_args()

    return args


def set_logger(args):
    """Set logs
    :param args: ArgumentParser
    """

    logformat = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    loglevel = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }[args.loglevel]
    if args.logfile:
        str_now = datetime.now().strftime("%Y%m%dT%H%M%S")
        logfile = os.path.join(args.outputdir, "logfile_" + str_now + ".log")
        logging.basicConfig(
            level=loglevel,
            format=logformat,
            handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
        )
    else:
        logging.basicConfig(
            level=loglevel,
            format=logformat,
            handlers=[logging.StreamHandler(sys.stdout)],
        )


def check_inputs(args):
    """Check input parameters
    :param args: ArgumentParser
    """

    # Check inputs
    if args.outputdir == ".":
        _logger.warning(
            "Output directory pass to argument does not exist, write log in current directory"
        )

    if args.watermask_tif is None:
        str_err = "Missing input arguments for single scene processing. watermask:{}".format(
            args.watermask_tif
        )
        raise ValueError(str_err)

    if args.reaches_shp is None:
        str_err = "Missing input arguments for single scene processing. reaches_shp:{}".format(
            args.reaches_shp
        )
        raise ValueError(str_err)

    if args.nodes_shp is None:
        str_err = "Missing input arguments for single scene processing. nodes_shp:{}".format(
            args.nodes_shp
        )
        raise ValueError(str_err)


def process_single_scene(
    str_watermask_tif=None,
    str_scene_datetime=None,
    str_reaches_shp=None,
    str_nodes_shp=None,
    flt_factor_width=10.0,
    str_outputdir=None,
    str_type_clean=None,
    str_type_label=None,
    more_outputs=False,
):
    """Process a reference watermask and compute river widths

    :param str_watermask_tif: str
        Full path to watermask from which compute river width
    :param str_scene_datetime: str
        Watermask scene datetime with format '%Y%m%dT%H%M%S'
    :param str_reaches_shp: str
        Full path to shapefile with river centerlines geometries
    :param str_nodes_shp: str
        Full path to shapefile with river nodes at which compute cross-sections
    :param flt_factor_width: float
        Base width multiplying factor to draw section
    :param str_outputdir: str
        Full path to directory where store outputs
    :param str_type_clean: str
        "base"/"waterbodies" : watermask cleaning method
    :param str_type_label: str
        "base" : watermask segmentation/labelling method
    :param more_outputs: bool
        If True, store all detailed outputs
    """

    _logger.info("=== Processing watermask: " + str_watermask_tif + " === : start\n")

    # Watermask filename to process
    if not os.path.isfile(str_watermask_tif):
        _logger.error(
            "Watermask file '{}' seems not to exist..".format(str_watermask_tif)
        )
    # str_scn_name = os.path.basename(str_watermask_tif).split(".")[0]

    # Width processing
    try:

        # Instanciate width processing class
        obj_widthprocessor = WidthProcessor(
            str_watermask_tif=str_watermask_tif,
            str_datetime=str_scene_datetime,
            str_reaches_shp=str_reaches_shp,
            str_nodes_shp=str_nodes_shp,
        )

        obj_widthprocessor.preprocessing(flt_factor_width)

        # Processing
        str_fpath_wm_tif, _ = obj_widthprocessor.processing(
            out_dir=str_outputdir,
            str_type_clean=str_type_clean,
            str_type_label=str_type_label,
        )

        # # PostProcessing
        obj_widthprocessor.postprocessing(str_outputdir, more_outputs)
        if not more_outputs:
            if os.path.exists(str_fpath_wm_tif):
                os.remove(str_fpath_wm_tif)

    except Exception as err:
        _logger.info(err)
        _logger.info("===> Fail during working with WidthProcessor object\n")
        pass


def main():
    """Main run"""

    # Parse inputs
    args = parse_inputs()

    # Set logs
    set_logger(args)

    # Check input parameters
    check_inputs(args)

    # Set output directory
    os.makedirs(args.outputdir, exist_ok=True)

    # Extract widths from a single SurfWater scene
    process_single_scene(
        str_watermask_tif=args.watermask_tif,
        str_scene_datetime=args.scn_datetime,
        str_reaches_shp=args.reaches_shp,
        str_nodes_shp=args.nodes_shp,
        str_outputdir=args.outputdir,
        str_type_clean=args.type_clean,
        str_type_label=args.type_label,
        flt_factor_width=args.factor_width,
        more_outputs=args.more_outputs,
    )


if __name__ == "__main__":
    main()
