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

"""
mirrowrsprocessor.py
: From an external watermask + a (set of) river centerline(s) and associated node along it (stations or calculus points),
derive a width estimation at said-nodes observed within the mask
"""

import logging
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
from pyproj import CRS
from shapely.geometry import MultiPolygon

from mirrowrs.constants import (FLT_LABEL_MAX_DIST, FLT_TOL_DIST_DEFAULT,
                                FLT_TOL_LEN_DEFAULT)
from mirrowrs.constants import L_WM_CLEAN_ALGO, L_WM_LABEL_ALGO
from mirrowrs.sections_reduction import reduce_section
from mirrowrs.tools import DisjointBboxError
from mirrowrs.watermask import WaterMask
from mirrowrs.widths import compute_widths_from_single_watermask

_logger = logging.getLogger("mirrowrs_processormodule")


class MIRROWRSPorcessor:
    _logger = logging.getLogger("mirrowrs_processormodule.MIRROWRSPorcessor")

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

        # Set _logger
        _logger = logging.getLogger("mirrowrs_processormodule.MIRROWRSPorcessor constructor")

        # Check inputs
        if str_watermask_tif is None:
            raise ValueError("Missing watermask GeoTiff input file")
        if not os.path.isfile(str_watermask_tif):
                raise FileExistsError("Input watermask GeoTiff does not exist")

        if gdf_sections is None:
            raise ValueError("Missing cross-section geometries inputs")
        if not isinstance(gdf_sections, gpd.GeoDataFrame):
            raise TypeError("Input cross_section geometries must be stored in a geopandas.GeoDataFrame")

        if gdf_reaches is None:
            raise ValueError("Missing reaches geometries")
        if not isinstance(gdf_reaches, gpd.GeoDataFrame):
            raise TypeError("Input reach geometries must be stored in a geopandas.GeoDataFrame")

        if not isinstance(str_provider, str):
            raise TypeError("Input watermask 'provider' attribute must be a string")

        if str_proj not in ["proj", "lonlat"]:
            raise NotImplementedError(
                "coordsyst available options are 'proj' or 'lonlat'")

        # Set attributes from inputs
        self.f_watermask_in = str_watermask_tif
        self.provider = str_provider
        self.proj = str_proj
        self.gdf_sections = gdf_sections
        self.gdf_reaches = gdf_reaches
        self.attr_reachid = attr_reachid

        # Additional checks
        if self.attr_reachid not in self.gdf_reaches.columns:
            raise ValueError("Input attr_reachid not in reach dataset columns")

        # Derived attributes
        self.scene_name = os.path.basename(self.f_watermask_in).split(".")[0]

        # Additionnal attributes computed later
        self.watermask = None
        self.scene_datetime = None

        # Set datetime information if provided
        if str_datetime is not None:
            self.scene_datetime = str_datetime
            try:
                _ = datetime.strptime(self.scene_datetime, "%Y%m%dT%H%M%S")
            except ValueError:
                raise ValueError(
                    "input datetime {} does not match expected format '%Y%m%dT%H%M%S'.".format(
                        self.scene_datetime
                    )
                )

        # Processing default values
        self.dct_cfg = {
            "clean": {
                "bool_toclean": True,
                "type_clean": "base",
                "fpath_wrkdir": ".",
                "gdf_waterbodies": None,
            },
            "label": {"bool_tolabel": False, "type_label": "base", "fpath_wrkdir": "."},
            "reduce": {
                "how": "simple",
                "attr_locxs": "loc_xs",
                "attr_nb_chan_max": "n_chan_max",
                "attr_nodepx": "proj_x",
                "attr_nodepy": "proj_y",
            },
            "widths": {"scenario": 0},
        }

    def check_bbox_compatibility(self):
        """Check if sections and watermask are spatially compatible"""

        # Set _logger
        _logger.info(
            "Checking bbox compatibility between watermask and river geometries"
        )

        # Project sections to WGS84
        if self.gdf_sections.crs != CRS(4326):
            gdf_crs_sections = self.gdf_sections.to_crs(CRS(4326))
        else:
            gdf_crs_sections = self.gdf_sections.copy(deep=True)

        # Extract sections bounding box
        bbox_sections = gdf_crs_sections.total_bounds
        polygon_sections = shapely.geometry.box(
            *bbox_sections
        )  # Convert bbox to Polygon object
        # Shapely 2.0 : box(xmin, ymin, xmax, ymax, ccw=True, **kwargs)

        # Extract watermask bounding box
        bbox_watermask = self.watermask.get_bbox()  # Load watermask bbox in lonlat
        polygon_watermask = shapely.geometry.box(
            *bbox_watermask
        )  # Convert watermask bbox to Polygon object

        if polygon_watermask.disjoint(polygon_sections):
            raise DisjointBboxError("Input watermask and input cross-sections are incompatible.")

        _logger.info(
            "Watermask and cross-sections are spatially compatible : proceed."
        )

    def preprocessing(self):
        """Preprocessing: load watermask, reproject sections et check bounding boxes intersections"""

        # Set _logger
        _logger.info("mirrowrs_processormodule.MIRROWRSPorcessor.preprocessing : Start")

        # Load WaterMask object
        self.watermask = WaterMask.from_tif(
            self.f_watermask_in, self.provider, self.proj
        )
        _logger.info("Watermask loaded..")

        # Reproject sections to watermask coordinate system
        self.gdf_reaches = self.gdf_reaches.to_crs(self.watermask.crs_epsg)
        self.gdf_sections = self.gdf_sections.to_crs(self.watermask.crs_epsg)
        _logger.info("Reaches and corss-section reprojected onto watermask coordinate system..")

        # Check boundingbox compatibility
        self.check_bbox_compatibility()

        _logger.info("mirrowrs_processormodule.MIRROWRSPorcessor.preprocessing : Done")

    def _read_cfg(self, dct_cfg=None):
        """Add default value to dct_cfg if keywords are missing

        :param dct_cfg: dict
            Input configuration dictionary
        :return dct_cfg: dict:
            Updated/Filled configuration dictionary
        """

        for key in ["clean", "label", "widths"]:
            if key not in dct_cfg.keys():
                dct_cfg[key] = self.dct_cfg[key]

            else:
                for subkey in self.dct_cfg[key].keys():
                    if subkey not in self.dct_cfg[key].keys():
                        dct_cfg[key][subkey] = self.dct_cfg[key][subkey]
        return dct_cfg

    def processing(self, dct_cfg=None):
        """Processing : extraction of widths from watermask

        :param dct_cfg: dict
            Processing configuration
            { "clean" : { "bool_clean" : True/False,
                      "type_clean" : base/waterbodies,
                      "fpath_wrkdir" : "."
                      "gdf_waterbodies" : gdf with polygon waterbodies to clean waterbodies [optionnal]
                    },
            "label" : { "bool_label" : True/False,
                      "type_label" : base,
                      "fpath_wrkdir" : "."
                    },
            "widths" : { scenario : 0/1/10/11
                     }
        }
        """

        # Set _logger
        _logger.info("mirrowrs_processormodule.MIRROWRSPorcessor.processing : Start")

        # Check cfg
        dct_cfg = self._read_cfg(dct_cfg)

        # Clean watermask
        _logger.info("Start to clean watermask..")
        try:
            if dct_cfg["clean"]["bool_clean"]:
                self.clean_watermask(dct_cfg)
        except Exception as err:
            _logger.info("Error while cleaning watermask")
            raise Exception(err)
        _logger.info("Clean watermask done..")

        # Label watermask
        _logger.info("Start to label watermask..")
        try:
            if dct_cfg["label"]["bool_label"]:
                self.label_watermask(dct_cfg)
        except Exception as err:
            _logger.info("Error while labelling watermask")
            raise Exception(err)
        _logger.info("Label watermask done..")

        _logger.info("mirrowrs_processormodule.MIRROWRSPorcessor.processing : Done")

    def clean_watermask(self, dct_cfg=None):
        """Clean watermask from non-river waterbodies

        :param dct_cfg: dict
            Processing configuration
        """

        # Check inputs
        if not dct_cfg["clean"]["type_clean"] in L_WM_CLEAN_ALGO:
            raise NotImplementedError("Input clean method is not implemented yet..")

        str_type_clean = dct_cfg["clean"]["type_clean"]
        _logger.info(f"Cleaning watermask - method :: {str_type_clean}")

        # Check config_dct
        if dct_cfg is None:
            dct_cfg = self.dct_cfg

        # Gather reaches and project them into the watermask coordinate system
        gdf_reaches_proj = self.gdf_reaches.to_crs(epsg=self.watermask.crs_epsg)

        # Gather wm as polygons
        gdf_wm_polygons = self.watermask.get_polygons(
            bool_clean=False,
            bool_label=False,
            bool_exterior_only=False,
            bool_indices=True,
        )

        # Apply regular cleaning : keep all waterbodies that intersect the riverline
        gdf_join_wm_reaches = gpd.sjoin(
            left_df=gdf_wm_polygons,
            right_df=gdf_reaches_proj,
            how="inner",
            predicate="intersects",
        )
        npar_idx_pol_notclean = np.setdiff1d(
            np.unique(gdf_wm_polygons.index), np.unique(gdf_join_wm_reaches.index)
        )

        # Apply waterbodies-type cleaning if activated
        if (dct_cfg["clean"]["gdf_waterbodies"] is not None
            and str_type_clean == "waterbodies"):

            # Check inputs
            if not isinstance(dct_cfg["clean"]["gdf_waterbodies"], gpd.GeoDataFrame):
                raise TypeError("Reference waterbodies must be provided as a geopandas.GeoDataFrame")
            gdf_tmp = dct_cfg["clean"]["gdf_waterbodies"].copy()
            gdf_waterbodies_wrk = gdf_tmp.to_crs(gdf_wm_polygons.crs)
            del gdf_tmp

            # Apply "waterbody" cleaning : keep all waterbodies that intersect the reference bodies
            gdf_join_wm_waterbodies = gpd.sjoin(
                left_df=gdf_wm_polygons,
                right_df=gdf_waterbodies_wrk,
                how="inner",
                predicate="intersects",
            )

            npar_idx_notclean_wb = np.setdiff1d(
                np.unique(gdf_wm_polygons.index),
                np.unique(gdf_join_wm_waterbodies.index),
            )

            npar_idx_pol_notclean = np.intersect1d(
                npar_idx_pol_notclean, npar_idx_notclean_wb
            )

        # Get pixc indexes from polygon indexes
        gdfsub_notclean_wm_polygons = gdf_wm_polygons.loc[
            npar_idx_pol_notclean, :
        ].copy()
        l_idx_pixc_notclean = [
            element
            for list_ in gdfsub_notclean_wm_polygons["indices"].values
            for element in list_
        ]

        # Update clean flag in the watermask
        self.watermask.update_clean_flag(mask=l_idx_pixc_notclean)

    def label_watermask(self, dct_cfg):
        """Label watermask into individual regions associated to a unique reach each

        :param dct_cfg: : dict
            Processing configuration
        """

        # Check inputs
        if not dct_cfg["label"]["type_label"] in L_WM_LABEL_ALGO:
            raise NotImplementedError("Input label method is not implemented yet..")

        str_type_label = dct_cfg["label"]["type_label"]
        _logger.info(f"Labelling watermask - method :: {str_type_label}")

        if str_type_label == "base":
            self.label_watermask_base()

    def label_watermask_base(self):
        """Label watermask into individual regions associated to a unique reach each
        Base method : each watermask pixel is associated to the closest reach
        """

        # Gather reaches and project them into the watermask coordinate system
        gdf_reaches_proj = self.gdf_reaches.loc[
            :, [self.attr_reachid, "geometry"]
        ].to_crs(epsg=self.watermask.crs_epsg)

        # Associate each pixel from wm to the closest reach
        gdf_label = gpd.sjoin_nearest(
            left_df=self.watermask.gdf_wm_as_pixc,
            right_df=gdf_reaches_proj,
            max_distance=FLT_LABEL_MAX_DIST,
            how="inner",
            distance_col="dist",
        )
        dct_label_update = {
            key: list(group)
            for key, group in gdf_label.groupby(by="index_right").groups.items()
        }
        self.watermask.update_label_flag(dct_label_update)

    def _prepare_tol_inputs(
        self, str_input_name=None, obj_to_prepare=None, default_value=0.0):
        """Add parameters to constrain procedure used to reduce cross-sections to watermask

        :param str_input_name: str
            key pointing towards the parameter to prepare
        :param obj_to_prepare: dict
            dict with configuration parameter
        :param default_value:
            default value of parameter, if key str_input_name is not in obj_to_prepare dictionnary
        """

        if str_input_name is None:
            raise ValueError(
                "Missing input argument 'str_input_name' for method '_prepare_tol_inputs'"
            )
        if obj_to_prepare is None:
            raise ValueError(
                "Missing input argument 'obj_to_prepare' for method '_prepare_tol_inputs'"
            )

        try:

            if isinstance(obj_to_prepare[str_input_name], float):  # Unique value
                self.gdf_sections.insert(
                    loc=2, column=str_input_name, value=obj_to_prepare[str_input_name]
                )

            # To debug
            elif isinstance(
                obj_to_prepare[str_input_name], str
            ):  # Column name in self.gdf_sections (from inputs)

                self.gdf_sections.rename(
                    columns={obj_to_prepare[str_input_name]: str_input_name},
                    inplace=True,
                )

            else:
                raise TypeError(
                    "Unexpected type of config input', "
                    "got {}, expecting (int, str)".format(
                        obj_to_prepare[str_input_name].__class__
                    )
                )

        except KeyError:  # entry is not defined
            self.gdf_sections.insert(loc=2, column=str_input_name, value=default_value)

    def _reduce_sections_over_reach(
        self, reach_id=None, lin_reach=None, pol_region=None, dct_cfg=None
    ):
        """Reduce all sections over a single reach

        :param reach_id: (str,int)
            ID of current reach
        :param lin_reach: LineString
            geometry of the centerline/reach
        :param pol_region: (Polygon/MultiPolygon)
            geometry of the vectorized watermask associated to the current reach
        :param dct_cfg: dict
            configuration dictionnary
        :return gdfsub_sections_byreach_onregion: GeoDataFrame
            Subset of original cross-section : cross-section have been either removed or reduced to the watermask
        """

        _logger.info(f"Reduce sections over reach {reach_id}")

        # Extract the cross-sections associated to the current reach
        gdfsub_sections_byreach = self.gdf_sections[
            self.gdf_sections[self.attr_reachid] == reach_id
        ].copy(deep=True)
        _logger.info("Extract the cross-sections associated to the current reach ..done")

        # In sections subset, keep only sections that intersect current region
        ser_bool_intersects = gdfsub_sections_byreach["geometry"].intersects(pol_region)
        gdfsub_sections_byreach_onregion = gdfsub_sections_byreach[
            ser_bool_intersects
        ].copy(deep=True)
        _logger.info("In sections subset, keep only sections that intersect current region ..done")
        _logger.info(f"Number of sections to reduce : {len(gdfsub_sections_byreach_onregion)}")

        # For remaining stations/sections, reduce their geometry to within the current region
        if len(gdfsub_sections_byreach_onregion) > 0:

            try:
                gdfsub_sections_byreach_onregion["geometry"] = (
                    gdfsub_sections_byreach_onregion.apply(
                        lambda x: reduce_section(
                            how=dct_cfg["reduce"]["how"],
                            lin_long_in=x.geometry,
                            pol_in=pol_region,
                            lin_rch_in=lin_reach,
                            flt_section_xs_along_rch=x[dct_cfg["reduce"]["attr_locxs"]],
                            flt_node_proj_x=x[dct_cfg["reduce"]["attr_nodepx"]],
                            flt_node_proj_y=x[dct_cfg["reduce"]["attr_nodepy"]],
                            int_nb_chan_max=x[dct_cfg["reduce"]["attr_nb_chan_max"]],
                            flt_tol_len=x["flt_tol_len"],
                            flt_tol_dist=x["flt_tol_dist"],
                        ),
                        axis=1,
                    )
                )
                _logger.info("For remaining stations/sections, reduce their geometry to within the current region.. done")

            except KeyError:

                _logger.info(
                    "For remaining stations/sections, reduce their geometry to within the current region.. done")
                _logger.info(
                    "KeyError warning: Simplified method applied..")
                if dct_cfg["reduce"]["how"] != "simple":
                    _logger.warning(
                        "Warning: Missing inputs use basic reduction (= method 'simple')."
                    )
                gdfsub_sections_byreach_onregion["geometry"] = (
                    gdfsub_sections_byreach_onregion.apply(
                        lambda x: reduce_section(
                            how=dct_cfg["reduce"]["how"],
                            lin_long_in=x.geometry,
                            pol_in=pol_region,
                            lin_rch_in=lin_reach,
                        ),
                        axis=1,
                    )
                )

            gdfsub_sections_byreach_onregion = gdfsub_sections_byreach_onregion[
                ~gdfsub_sections_byreach_onregion["geometry"].is_empty
            ]

            return gdfsub_sections_byreach_onregion

        else:
            _logger.warning("Warning: no sections intersect the region..")
            return None

    def reduce_sections(
        self,
        dct_cfg=None,
        flt_tol_len=FLT_TOL_LEN_DEFAULT,
        flt_tol_dist=FLT_TOL_DIST_DEFAULT,
    ):
        """Reduce sections geometry to its associated region

        :param dct_cfg: dict
            Processing configuration
        :param flt_tol_len: float
            Default value for minimal cross-section piece length
            when segmented over multiple channels
        :param flt_tol_dist: float
            Default value for minimum distance between adjacent pieces of sub-cross-section
            when segmented over multiple channels
        :return gdf_sections_out : gpd.GeoDataFrame
            Corrected cross-section geometries
        """

        # Check config_dct
        if dct_cfg is None:
            dct_cfg = self.dct_cfg

        if "reduce" not in dct_cfg.keys():
            dct_cfg["reduce"] = self.dct_cfg["reduce"]

        if dct_cfg["reduce"]["how"] != "simple":
            # Prepare input flt_tol_len
            self._prepare_tol_inputs(
                str_input_name="flt_tol_len",
                obj_to_prepare=dct_cfg["reduce"],
                default_value=flt_tol_len,
            )

            # Prepare input flt_tol_dist
            self._prepare_tol_inputs(
                str_input_name="flt_tol_dist",
                obj_to_prepare=dct_cfg["reduce"],
                default_value=flt_tol_dist,
            )

        gdf_wm_labelled_pol = self.watermask.get_polygons(
            bool_clean=dct_cfg["clean"]["bool_clean"],
            bool_label=dct_cfg["label"]["bool_label"],
            bool_indices=False,
            bool_exterior_only=False,
        )
        _logger.info("Polygons retrieved")

        #TODO : case when bool_label is False

        l_gdfsub_sections = []
        for label, group in gdf_wm_labelled_pol.groupby(by="label").groups.items():

            # Extract watermask region associated to current reach
            pol_region = MultiPolygon(
                gdf_wm_labelled_pol.loc[group, "geometry"].tolist()
            )

            # Check if reach has not been previously cleaned
            if label not in self.gdf_reaches.index:
                _logger.info(f"Reach with label {label} has been removed.. ignored")
                gdfsub_sections_byreach_onregion = None

            else:
                # Extract sections associated to unique current reach
                reach_id = self.gdf_reaches.at[label, self.attr_reachid]
                lin_reach = self.gdf_reaches.at[label, "geometry"]

                # Reduce section
                gdfsub_sections_byreach_onregion = self._reduce_sections_over_reach(
                    reach_id=reach_id,
                    lin_reach=lin_reach,
                    pol_region=pol_region,
                    dct_cfg=dct_cfg,
                )
                _logger.info(f"Sections reduced over reach {reach_id} ..done")

            if gdfsub_sections_byreach_onregion is not None:
                # Add label associated to sections over the current reach
                gdfsub_sections_byreach_onregion.insert(2, "label", int(label))

                # Add reduced section to the full set
                l_gdfsub_sections.append(gdfsub_sections_byreach_onregion)

        # Gather all sections
        gdf_sections_out = pd.concat(l_gdfsub_sections)

        return gdf_sections_out

    def postprocessing(self, dct_cfg=None, str_fpath_dir_out=".", str_fpath_wm_in=None):
        """Post-processing :: Reduce cross-sections and compute node-scale widths

        :param str_fpath_wm_in: str
            Full path to prepared watermask
        :param dct_cfg: dict
            Processing configuration
        :param str_fpath_dir_out: str
            Full path to directory where to save outputs and intermediate files
        :return gdf_widths: gpd.GeoDataFrame
            Node-scale widths
        :return str_fpath_wm_tif: str
            Full path to prepared watermask
        """

        # Set _logger
        _logger.info("mirrowrs_processormodule.MIRROWRSPorcessor.postprocessing : Start")

        # Prepare sections
        gdf_wrk_sections = self.reduce_sections(dct_cfg)
        _logger.info("Sections reduced..")

        # Process width
        if str_fpath_wm_in is None:
            str_fpath_wm_tif = self.watermask.save_wm(
                fmt="tif",
                bool_clean=dct_cfg["clean"]["bool_clean"],
                bool_label=dct_cfg["label"]["bool_label"],
                str_fpath_dir_out=str_fpath_dir_out,
                str_suffix="readytouse",
            )
        else:
            str_fpath_wm_tif = str_fpath_wm_in

        with rio.open(str_fpath_wm_tif) as src:
            gdf_widths, _ = compute_widths_from_single_watermask(
                scenario=dct_cfg["widths"]["scenario"],
                watermask=src,
                sections=gdf_wrk_sections,
                buffer_length=8.0 * self.watermask.res,
                label_attr="label",
            )
        _logger.info("Width computed..")

        _logger.info("mirrowrs_processormodule.MIRROWRSPorcessor.postprocessing : Done")

        return gdf_widths, str_fpath_wm_tif
