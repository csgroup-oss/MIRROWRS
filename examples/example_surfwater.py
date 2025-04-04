# Copyright (C) 2023-2024 CS GROUP France, https://csgroup.eu
# Copyright (C) 2024 CNES.
#
# This file is part of BAS (Buffer Around Sections)
#
#     https://github.com/CS-SI/BAS
#
# Authors:
#     Charlotte Emery
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

# More complete example use of BAS tools:
# - How to add a new/specific type of watermask via child classes WaterMaskCE and BASProcessorCE
# - How to compute a width product as a combination of BAS-based widths products (from scenario 0 and 11)
# - A first uncertainty model for node-scale width

import logging
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from bas.basprocessor import BASProcessor
from bas.rivergeomproduct import RiverGeomProduct
from bas.tools import FileExtensionError
from bas.watermask import WaterMask

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
    LOGGER = logging.getLogger("BAS PROCESSING")
    LOGGER.info("Get width_1-related info : width_1 + beta")
    gdf_widths_out = gdf_widths_ortho.copy(deep=True)
    # Width from ortho sections :: width_1
    gdf_widths_out.rename(mapper={"width": "width_1"}, axis=1, inplace=True)

    LOGGER.info("Get width_2-related info : width_2 + theta")
    # Width from parallel sections : width_2
    gdf_widths_out.insert(
        loc=len(gdf_widths_out.columns) - 1,
        column="width_2",
        value=gdf_widths_chck["width"],
    )

    # Add theta and sin_theta columns to output width dataframe
    # Theta is an angle in degree
    LOGGER.info("Set theta components")
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
    gdf_widths_out["cos_theta"] = gdf_widths_chck["theta"].apply(lambda theta_deg : np.cos(theta_deg*np.pi/180.))

    # Add final width product column
    LOGGER.info("Set width")
    gdf_widths_out.insert(
        loc=len(gdf_widths_out.columns) - 1, column="width", value=np.nan
    )

    # beta = 0 OR W2 == nan : W = W1
    # If no intersection between base section and the neighbouring one, check section is irrelevant
    # => beta=0 + no contribution from W2
    ser_mask_beta = gdf_widths_out["beta"].apply(lambda beta: True if beta==0 else False)
    ser_mask_w2 = gdf_widths_out["width_2"].apply(lambda w2: True if np.isnan(w2) else False)
    ser_mask = ser_mask_beta | ser_mask_w2
    gdf_widths_out.loc[ser_mask, "width"] = gdf_widths_out.loc[ser_mask, "width_1"]

    # Else beta>0 AND W2!=nan : W = (W1 + |cos(theta)|xW2)/(1+|cos(theta)|)
    ser_cos_theta_tmp = gdf_widths_out["cos_theta"].apply(np.abs)
    gdf_widths_out.loc[~ser_mask, "width"] = gdf_widths_out.loc[~ser_mask, "width_1"] + gdf_widths_out.loc[~ser_mask, "width_2"].mul(ser_cos_theta_tmp)
    gdf_widths_out.loc[~ser_mask, "width"] = gdf_widths_out.loc[~ser_mask, "width"].div(ser_cos_theta_tmp[~ser_mask].apply(lambda ct: ct + 1.0))

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

        Parameters
        ----------
        surfwater_tif : str
            Full path to surfwater mask stored in a GeoTiff file
        """
        LOGGER = logging.getLogger("BAS PROCESSING")
        # Instanciate object
        klass = WaterMaskCHM()
        klass.str_provider = "SurfWater"
        klass.str_fpath_infile = surfwater_tif

        # Set watermask rasterfile
        if not os.path.isfile(surfwater_tif):
            LOGGER.error("Watermask.from_surfwater: Input tif file does not exist..")
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

            band = src.read(1)
            _ = klass.band_to_pixc(band, src, exclude_values=0)

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
        LOGGER = logging.getLogger("BAS PROCESSING")
        print("----- WidthProcessing = Preprocessing -----")
        print("")

        # Load WaterMask object
        if bool_load_wm:
            if self.provider == "SW":  # Watermask is provided by Surfwater
                LOGGER.info("Load watermask from Surfwater")
                self.watermask = WaterMaskCHM.from_surfwater(self.f_watermask_in)

                # Check boundingbox compatibility
                self.check_bbox_compatibility()
            else:
                LOGGER.error(f"Provider {self.provider} not recognized")
                raise NotImplementedError
            LOGGER.info("Watermask loaded..")
        else:
            self.watermask = WaterMaskCHM()
            LOGGER.info("User chose not to load watermask..")

        # Reproject sections to watermask coordinate system
        LOGGER.info("Reproject sections to watermask coordinate system")
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
        LOGGER.info("Reproject sections to watermask coordinate system done ..")

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
        """Class constructor"""
        LOGGER = logging.getLogger("BAS PROCESSING")

        LOGGER.info("Instanciate WidthProcessor")

        if str_watermask_tif is not None:
            self.f_watermask_in = str_watermask_tif
            if not os.path.isfile(str_watermask_tif):
                LOGGER.error("Input watermask GeoTiff does not exist")
                raise FileExistsError("Input watermask GeoTiff does not exist")
        else:
            LOGGER.error("Missing watermask GeoTiff input file")
            raise ValueError("Missing watermask GeoTiff input file")
        self.scene_name = os.path.basename(self.f_watermask_in).split(".")[0]

        if str_datetime is not None:
            self.scene_datetime = str_datetime
            try:
                _ = datetime.strptime(self.scene_datetime, "%Y%m%dT%H%M%S")
            except ValueError:
                LOGGER.error(
                    "input datetime {} does not match format '%Y%m%dT%H%M%S'.".format(
                        self.scene_datetime
                    )
                )
                raise ValueError(
                    "input datetime {} does not match format '%Y%m%dT%H%M%S'.".format(
                        self.scene_datetime
                    )
                )
        else:
            LOGGER.error("Missing scene datetime information input")
            raise ValueError("Missing scene datetime information input")

        if str_reaches_shp is not None:
            if not os.path.isfile(str_reaches_shp):
                LOGGER.error("Input reaches shapefile does not exist")
                raise FileExistsError("Input reaches shapefile does not exist")
            self.reaches_shp = str_reaches_shp
            self.gdf_reaches = gpd.read_file(self.reaches_shp)
        else:
            LOGGER.error("Input reaches shapefile does not exist")
            raise FileExistsError("Input reaches shapefile does not exist")

        if str_nodes_shp is not None:
            if not os.path.isfile(str_nodes_shp):
                LOGGER.error("Input nodes shapefile does not exist")
                raise FileExistsError("Input nodes shapefile does not exist")
            self.nodes_shp = str_nodes_shp
            self.gdf_nodes = gpd.read_file(self.nodes_shp)
        else:
            LOGGER.error("Input nodes shapefile does not exist")
            raise FileExistsError("Input nodes shapefile does not exist")

        self.gdf_sections_ortho = None
        self.gdf_sections_chck = None

        self.bas_processor_o = None
        self.bas_processor_c = None

        self.gdf_nodescale_widths = None

    def preprocessing(self, flt_factor_width=10.0):
        """Prepare SWOT-like product watermask to process and associated SWORD nodes and reaches"""

        LOGGER = logging.getLogger("WidthProcessing.preprocessing")

        with rio.open(self.f_watermask_in) as src:
            crs_wm_in = src.crs

        # Instanciate RiverGeom object
        LOGGER.info("Instanciate RiverGeomProduct object ..")
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
                crs_in=crs_wm_in
            )
            LOGGER.info("Instanciation done..")
        except Exception as err:
            LOGGER.error(err)
            LOGGER.error("Instanciate RiverGeomProduct object KO ..")
            raise Exception

        # Set centerlines for section definition
        LOGGER.info("Set centerlines for section definition")
        try:
            obj_rivergeom.draw_allreaches_centerline()
        except Exception as err:
            LOGGER.error(err)
            LOGGER.error("Set centerlines for section definition KO ..")
            raise Exception

        # Traditionnal orthogonal sections
        LOGGER.info("Traditionnal orthogonal sections ..")
        try:
            self.gdf_sections_ortho = obj_rivergeom.draw_allreaches_sections(
                type="ortho", flt_factor_width=flt_factor_width
            )
        except Exception as err:
            LOGGER.error(err)
            LOGGER.error("Draw orthogonal sections KO ..")
            raise Exception

        # Checking parallel sections
        LOGGER.info("Parallel sections given the main direction of the reach")
        try:
            self.gdf_sections_chck = obj_rivergeom.draw_allreaches_sections(
                type="chck", flt_factor_width=flt_factor_width
            )
        except Exception as err:
            LOGGER.error(err)
            LOGGER.error("Draw parallel sections KO ..")
            raise Exception

        # Instanciate BASProcessorCalVal objects
        try:
            LOGGER.info("Instanciate BASProcessor object for sections_ortho")
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
            LOGGER.error(err)
            LOGGER.error("Instanciate width processor ortho KO ..")
            raise Exception

        try:
            LOGGER.info("Instanciate BASProcessor object for sections_chck")
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
            LOGGER.error(err)
            LOGGER.error("Instanciate width processor check KO ..")
            raise Exception

    def basprocessing_ortho(
        self, out_dir=".", str_pekel_shp=None, str_type_clean=None, str_type_label=None
    ):
        """Perform BASProcessor processing (=clean+label) - only on BASProcessor associated to ortho section

        :param out_dir:
        :param str_pekel_shp:
        :param str_type_clean:
        :param str_type_label:
        :return:
        """
        LOGGER = logging.getLogger("BAS PROCESSING")
        LOGGER.info("Processing based on orthogonal sections")
        try:
            dct_cfg_o = DCT_CONFIG_O

            # Set cleaning and labelling method
            LOGGER.info(f"Set cleaning method to : {str_type_clean}")
            if str_type_clean is not None:
                if str_type_clean not in ["base", "waterbodies"]:
                    LOGGER.info(
                        "Undefined cleaning type .. Ignore .. use default value"
                    )
                else:
                    dct_cfg_o["clean"]["type_clean"] = str_type_clean

            LOGGER.info(f"Set labelling method to : {str_type_label}")
            if str_type_label is not None:
                if str_type_label not in ["base"]:
                    LOGGER.info(
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
            LOGGER.error(err)
            LOGGER.error("Processing (clean+label) based on orthogonal section KO ..")
            raise Exception

    def basproccessing_ortho_widths(self, out_dir="."):
        LOGGER = logging.getLogger("BAS PROCESSING")
        LOGGER.info("Width computation based on orthogonal sections")
        try:
            dct_cfg_o = DCT_CONFIG_O

            gser_proj_nodes = self.gdf_nodes["geometry"].to_crs(
                self.bas_processor_o.watermask.crs
            )

            # Add required attributes for sections reduction
            attr_nodepx = dct_cfg_o["reduce"]["attr_nodepx"]
            self.bas_processor_o.gdf_sections.insert(
                loc=3, column=attr_nodepx, value=0.0
            )
            self.bas_processor_o.gdf_sections[attr_nodepx] = gser_proj_nodes.loc[
                self.bas_processor_o.gdf_sections.index
            ].x

            attr_nodepy = dct_cfg_o["reduce"]["attr_nodepy"]
            self.bas_processor_o.gdf_sections.insert(
                loc=4, column=attr_nodepy, value=0.0
            )
            self.bas_processor_o.gdf_sections[attr_nodepy] = gser_proj_nodes.loc[
                self.bas_processor_o.gdf_sections.index
            ].y

            attr_n_chan_max = dct_cfg_o["reduce"]["attr_nb_chan_max"]
            self.bas_processor_o.gdf_sections.insert(
                loc=5, column=attr_n_chan_max, value=0
            )
            self.bas_processor_o.gdf_sections[attr_n_chan_max] = self.gdf_nodes.loc[
                self.bas_processor_o.gdf_sections.index, attr_n_chan_max
            ]

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

            gdf_widths_ortho, str_fpath_wm_out = self.bas_processor_o.postprocessing(
                dct_cfg=dct_cfg_o, str_fpath_dir_out=out_dir
            )

            return gdf_widths_ortho, str_fpath_wm_out

        except Exception as err:
            LOGGER.error(err)
            LOGGER.error("Processing based on orthogonal section KO ..")
        raise Exception

    def basprocessing_chck(self):
        LOGGER = logging.getLogger("BAS PROCESSING")
        LOGGER.info("Processing based on paralell sections: preparation")
        try:

            self.bas_processor_c.watermask = self.bas_processor_o.watermask

        except Exception as err:
            LOGGER.error(err)
            LOGGER.error("Processing based on parallel section : preparation KO ..")
            raise Exception

    def basprocessing_chck_widths(self, str_fpath_wm_in=None):
        LOGGER = logging.getLogger("BAS PROCESSING")
        LOGGER.info("Width computation based on check sections")
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
            LOGGER.error(err)
            LOGGER.error("Processing based on check section KO ..")
        raise Exception

    def processing(
        self, out_dir=".", str_pekel_shp=None, str_type_clean=None, str_type_label=None
    ):
        """Produce riverwidth derived from watermask"""
        LOGGER = logging.getLogger("BAS PROCESSING")
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
        print(gdf_widths_chck)

        # Compute node-scale widths
        LOGGER.info("Compute widths at node scale")
        try:
            self.gdf_nodescale_widths = compute_nodescale_width(
                gdf_widths_ortho, gdf_widths_chck
            )

            self.gdf_nodescale_widths.insert(loc=2, column="provider", value="SW")
            self.gdf_nodescale_widths.insert(loc=3, column="bool_ko", value=0)
            self.gdf_nodescale_widths["bool_ko"] = self.gdf_nodescale_widths[
                "width"
            ].apply(lambda w: np.logical_or(np.isnan(w), w == 0))

            LOGGER.info("Node-scale widths computed..")
        except Exception as err:
            LOGGER.error(err)
            LOGGER.info("Node-scale widths computed ko..")
            raise Exception

        # Compute node-scale width errors
        LOGGER.info("Compute width error at node scale")
        try:
            ser_errtot, ser_sigo, ser_sigr, ser_sigs = compute_nodescale_widtherror(
                self.gdf_nodescale_widths, self.bas_processor_o.watermask.res
            )
            self.gdf_nodescale_widths.insert(loc=3, column="width_u", value=ser_errtot)
            self.gdf_nodescale_widths.insert(loc=4, column="sigo", value=ser_sigo)
            self.gdf_nodescale_widths.insert(loc=5, column="sigr", value=ser_sigr)
            self.gdf_nodescale_widths.insert(loc=6, column="sigs", value=ser_sigs)

            LOGGER.info("Node-scale width errors computed..")
        except Exception as err:
            LOGGER.error(err)
            LOGGER.info("Node-scale width errors computed ko..")
            raise Exception

        return str_fpath_wm_tif, str_fpath_wm_out

    def postprocessing(self, output_dir=".", more_outputs=False):
        """Save processed riverwidths into files"""

        LOGGER = logging.getLogger("WidthProcessing.postprocessing")

        LOGGER.info("Save node-scale width as csv ")
        df_nodes_width = self.gdf_nodescale_widths.loc[
            :, ["reach_id", "node_id", "provider", "width", "width_u"]
        ].copy(deep=True)
        df_nodes_width = df_nodes_width.dropna()
        df_nodes_width["datetime"] = self.scene_datetime
        width_nodes_csv = self.scene_name + "_nodescale_BAS_widths.csv"
        df_nodes_width.to_csv(os.path.join(output_dir, width_nodes_csv))
        LOGGER.info("Node-scale width saved to csv..")

        # Save cross-sections with associated width as shp in epsg:4326
        if more_outputs:
            LOGGER.info(
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
            LOGGER.info("Node-scale width saved to shp..")


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
    LOGGER = logging.getLogger("BAS PROCESSING")
    LOGGER.info("=== Processing watermask: " + str_watermask_tif + " === : start\n")

    # watermask filename to process
    if not os.path.isfile(str_watermask_tif):
        LOGGER.error(
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
        LOGGER.info(err)
        LOGGER.info("===> Fail during working with WidthProcessor object\n")
        pass


def main():
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
        type=str.lower,
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

    os.makedirs(args.outputdir, exist_ok=True)

    # Set logs
    logformat = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    LOGGER = logging.getLogger("BAS PROCESSING")
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

    # Set output dir
    if args.outputdir == ".":
        LOGGER.warning(
            "Output directory pass to argument does not exist, write log in current directory"
        )

    if args.watermask_tif is None or args.reaches_shp is None or args.nodes_shp is None:
        str_err = "Missing one or more input arguments for single scene processing. watermask:{}, reaches_shp:{}, nodes_shp:{}".format(
            args.watermask_tif, args.reaches_shp, args.nodes_shp
        )
        LOGGER.error(str_err)
        raise ValueError(str_err)

    LOGGER.info(f"More outputs: {args.more_outputs}")
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
