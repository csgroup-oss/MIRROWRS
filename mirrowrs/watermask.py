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
module watermask.py
: Contains classes to manipulate watermask raster
"""

import os
import geopandas as gpd
import numpy as np
import numpy.ma as ma
from osgeo import osr
import pandas as pd
import rasterio as rio
from rasterio.features import shapes
from shapely.geometry import shape, Point, Polygon
import logging

from mirrowrs.tools import FileExtensionError, DimensionError


_logger = logging.getLogger("watermask_module")


class WaterMask:

    def __init__(self):
        """Class constructor
        """

        self.str_provider = None
        self.str_fpath_infile = None

        self.bbox = None
        self.crs = None
        self.crs_epsg = None
        self.coordsyst = None  # "proj" or "lonlat"

        self.transform = None
        self.width = None
        self.height = None
        self.dtypes = None
        self.nodata = None
        self.res = None

        self.gdf_wm_as_pixc = None
        self.dtype_label_out = None

    @classmethod
    def from_tif(cls, watermask_tif=None, str_origin=None, str_proj="proj"):
        """Alternate constructor from any GeoTiff file

        :param watermask_tif: str
            Path to GeoTiff file containting watermask
        :param str_origin: str
            Indicates origin of watermask
        :param str_proj: str
            ["proj", "lonlat"]
        :return klass: WaterMask
        """

        klass = WaterMask()

        # Set watermask origin (can be anything)
        klass.str_provider = str_origin

        # Set watermask rasterfile
        if not os.path.isfile(watermask_tif):
            raise FileExistsError("Input watermak_tif file does not exist..")
        else:
            if not watermask_tif.endswith(".tif"):
                raise FileExtensionError(message="Input file is not a .tif")
        klass.str_fpath_infile = watermask_tif

        # Set raster coordinate system
        if str_proj not in ["proj", "lonlat"]:
            raise NotImplementedError("coordsyst available options are proj or lonlat")
        klass.coordsyst = str_proj

        # Set raster bounding box, crs and resolution
        with rio.open(watermask_tif, 'r') as src:
            klass.crs = src.crs
            klass.crs_epsg = src.crs.to_epsg()
            klass.bbox = (src.bounds.left,
                          src.bounds.bottom,
                          src.bounds.right,
                          src.bounds.top)

            klass.transform = src.transform
            klass.res = src.transform.a
            klass.width = src.width
            klass.height = src.height
            klass.nodata = src.nodata

            klass.dtypes = src.dtypes[0]
            klass.dtype_label_out = src.dtypes[0]

            band = src.read(1)
            _ = klass.band_to_pixc(band, src)

        return klass

    def band_to_pixc(self, npar_band, raster_src, exclude_values=None, **kwargs):
        """Transform the input raster band into a pixel-cloud like object for easier manipulation

        :param npar_band: np.array
            watermask data band
        :param raster_src: rasterio.io.Dataset
            watermask as raster
        :param exclude_values: (int, float)
            value in watermask to consider as nodata (to ignore)
        :return gdf_wm_as_pixc: gpd.GeoDataFrame
            watermask sorted as a pixel-cloud
        """

        # Chcek input npar_band
        if not isinstance(npar_band, np.ndarray):
            raise TypeError(f"Input npar_band must be of class np.ndarray, got {npar_band.__class__}")
        if npar_band.ndim != 2:
            raise DimensionError(f"Input npar_band has {npar_band.ndim} dimensions, expecting 2.")

        # Turn watermask band into a point-cloud format
        band_flat = npar_band.flatten()
        indices = np.where(band_flat != self.nodata)[0]
        if exclude_values is not None:
            if isinstance(exclude_values, (int, float)):
                indices_excluded = np.where(band_flat == exclude_values)[0]
                indices = np.setdiff1d(indices, indices_excluded)
            else:
                raise NotImplementedError("For now, can only exclude single value. If more, need to be implemented")

        l_index = [t for t in np.unravel_index(indices, npar_band.shape)]
        l_coords = [raster_src.xy(i, j) for i, j in
                    zip(np.unravel_index(indices, npar_band.shape)[0], np.unravel_index(indices, npar_band.shape)[1])]

        # Store watermask labels
        self.gdf_wm_as_pixc = gpd.GeoDataFrame(
            pd.DataFrame({"i": [i for i in l_index[0]],
                          "j": [j for j in l_index[1]],
                          "label": [band_flat[k] for k in indices],
                          "clean": np.ones(indices.shape, dtype=np.uint8)
                          },
                         index=pd.Index(indices)),
            geometry=gpd.GeoSeries(
                [Point(x, y) for (x, y) in l_coords],
                crs=raster_src.crs,
                index=pd.Index(indices)
            ),
            crs=raster_src.crs
        )

        return self.gdf_wm_as_pixc

    def __str__(self):
        """String method

        :return message: str
            Basic description of the watermask
        """

        if self.str_provider is not None:
            message = "Watermask product from {}\n".format(self.str_provider)
        else:
            message = "Watermask product from {}\n".format(self.str_fpath_infile)

        return message

    def get_bbox(self):
        """Derive bounding box of current WaterMask product in lon-lat system

        :return minlon: float
            minimum longitude
        :return minlat: float
            minimum latitude
        :return maxlon: float
            maximum longitude
        :return maxlat: float
            maximum latitude:
        """

        if self.coordsyst == "lonlat":

            minlon = self.bbox[0]
            minlat = self.bbox[1]
            maxlon = self.bbox[2]
            maxlat = self.bbox[3]

        elif self.coordsyst == "proj":

            src = osr.SpatialReference()
            src.ImportFromProj4(self.crs.to_proj4())

            tgt = osr.SpatialReference()
            tgt.ImportFromEPSG(4326)

            osr_transform = osr.CoordinateTransformation(src, tgt)

            lonlat_bottomleft_edge = osr_transform.TransformPoint(self.bbox[0], self.bbox[1])
            # output : (lat, lon, z)
            lonlat_topleft_edge = osr_transform.TransformPoint(self.bbox[0], self.bbox[3])
            lonlat_bottomright_edge = osr_transform.TransformPoint(self.bbox[2], self.bbox[1])
            lonlat_topright_edge = osr_transform.TransformPoint(self.bbox[2], self.bbox[3])

            minlon = min([lonlat_bottomleft_edge[1], lonlat_topleft_edge[1]])
            maxlon = max([lonlat_bottomright_edge[1], lonlat_topright_edge[1]])

            minlat = min([lonlat_bottomleft_edge[0], lonlat_bottomright_edge[0]])
            maxlat = max([lonlat_topleft_edge[0], lonlat_topright_edge[0]])


        else:
            raise NotImplementedError

        return minlon, minlat, maxlon, maxlat

    def get_band(self, bool_clean=True, bool_label=True, as_ma=True):
        """ Return wm as band-like format with activated flags

        :param bool_clean: bool
            If True, return cleaned watermask
        :param bool_label: bool
            If True, return labelled watermask
        :param as_ma : boolean
            If True, band is retrun as a masked array, else a simple np.array
        :return npar_band: np.ma.array or np.array
            Watermask band
        """

        npar_band_flat = np.ones((self.width * self.height,)) * self.nodata
        if bool_clean and bool_label:
            gdfsub_wrk = self.gdf_wm_as_pixc[self.gdf_wm_as_pixc["clean"] == 1]
            npar_band_flat[gdfsub_wrk.index] = gdfsub_wrk["label"]

        elif bool_clean and not bool_label:
            gdfsub_wrk = self.gdf_wm_as_pixc[self.gdf_wm_as_pixc["clean"] == 1]
            npar_band_flat[gdfsub_wrk.index] = 1

        elif not bool_clean and bool_label:
            npar_band_flat[self.gdf_wm_as_pixc.index] = self.gdf_wm_as_pixc["label"]

        else:
            npar_band_flat[self.gdf_wm_as_pixc.index] = 1

        if as_ma:
            npar_band = ma.array(
                npar_band_flat.reshape((self.height, self.width)),
                mask=npar_band_flat.reshape((self.height, self.width)) == self.nodata)
        else:
            npar_band = npar_band_flat.reshape((self.height, self.width))

        if not bool_label:
            npar_band = npar_band.astype(self.dtypes)

        else:
            npar_band = npar_band.astype(self.dtype_label_out)

        return npar_band

    def get_polygons(self, bool_clean=True, bool_label=True, bool_indices=True, bool_exterior_only=True):
        """ Turn watermask into a set of polygons given clean and label flags for vectorial studies

        :param bool_clean: boolean
            If True, return cleaned watermask
        :param bool_label: boolean
            If True, return labelled watermask
        :param bool_indices: boolean
            If True, return indices of pixels forming each polygons
        :param bool_exterior_only: boolean
            If True, compute only the exterior of each polygons
        :return gdf_wm_as_pol: gpd.GeoDataFrame
            Watermask as a set of polygons
        """

        npar_band = self.get_band(bool_clean, bool_label, as_ma=True)

        l_pol_wm = []
        l_pol_value = []
        for geom, value in shapes(npar_band.data,
                                  mask=(~npar_band.mask),
                                  transform=self.transform):

            # Get label
            l_pol_value.append(value)

            # Get geometry
            if not bool_exterior_only:
                pol_wm = shape(geom)
            else:
                pol_wm = Polygon(geom["coordinates"][0])
            l_pol_wm.append(pol_wm)

        gdf_wm_as_pol = gpd.GeoDataFrame(
            pd.DataFrame({"label": l_pol_value,
                          "clean": [1] * len(l_pol_value),
                          "indices": None}),
            geometry=gpd.GeoSeries(
                l_pol_wm, crs=self.crs
            ),
            crs=self.crs
        )

        if bool_indices:
            gdf_join = gpd.sjoin(left_df=self.gdf_wm_as_pixc,
                                 right_df=gdf_wm_as_pol,
                                 how="inner",
                                 predicate="within")
            for index_right, group in gdf_join.groupby(by="index_right").groups.items():
                gdf_wm_as_pol.at[index_right, "indices"] = list(group)

        return gdf_wm_as_pol

    def update_clean_flag(self, mask=None):
        """ Update clean flags: for input indexes in mask, turn clean flag to 0

        :param mask: iterable
            List of pixel indexes to set as "not-clean"
        """

        self.gdf_wm_as_pixc.loc[mask, "clean"] = 0

    def update_label_flag(self, dct_label=None, dtype_labelled=None):
        """ Update label values: for each pixels, associate the label of the watermask segmentation

        :param dct_label: dct
            Mapping between watermask segmentation label and pixels : {label: l_pixel_indices}
        """

        # Get max label value
        int_max_label = max(dct_label.keys())

        # Update label dtype if necassary
        if int_max_label >= 255:
            self.gdf_wm_as_pixc["label"] = self.gdf_wm_as_pixc["label"].astype(np.uint16)

        # update label flag
        for label, indices in dct_label.items():
            self.gdf_wm_as_pixc.loc[indices, "label"] = int(label)

        # update labelled watermask output dtype if necessary
        if dtype_labelled is not None:
            self.dtype_label_out = dtype_labelled

        else:
            int_max_label = self.gdf_wm_as_pixc["label"].max()
            if int_max_label < 255:
                self.dtype_label_out = rio.uint8
                self.nodata = 255
            else:
                self.dtype_label_out = rio.uint16
                self.nodata = 65535

    def save_wm(self, fmt="tif", bool_clean=True, bool_label=True, str_fpath_dir_out=".", str_suffix=None):
        """ Save the watermask in the asked format : tif/pixc/polygons + "clean/label"

        :param fmt: str
            Format in which save the watermask: ["tif", "pixc", "shp"]
        :param bool_clean: boolean
            If True, save clean version of watermask
        :param bool_label: boolean
            If True, save labelled version of watermask
        :param str_fpath_dir_out: str
            Full path to the directory where to save the watermask file
        :param str_suffix: str
            Suffix to append at the end of the watermask filename
        :return: str
            Full path to complete watermask filename
        """

        str_basename = self.str_fpath_infile.split("/")[-1].split(".")[0]
        if bool_clean:
            str_basename += "_clean"
        if bool_label:
            str_basename += "_label"
        if str_suffix is not None:
            str_basename += f"_{str_suffix}"

        if fmt == "tif":

            str_fpath_wm_out_tif = os.path.join(str_fpath_dir_out, str_basename + ".tif")
            npar_band_tosave = self.get_band(bool_clean=bool_clean, bool_label=bool_label, as_ma=False)

            with rio.open(
                    str_fpath_wm_out_tif,
                    mode="w",
                    driver="GTiff",
                    height=npar_band_tosave.shape[0],
                    width=npar_band_tosave.shape[1],
                    count=1,
                    dtype=self.dtype_label_out,
                    crs=self.crs,
                    transform=self.transform,
                    nodata=self.nodata
            ) as new_dataset:
                new_dataset.write(npar_band_tosave, 1)

            return str_fpath_wm_out_tif

        elif fmt == "pixc":

            str_fpath_wm_out_pixc_shp = os.path.join(str_fpath_dir_out, str_basename + "_pixc.shp")
            self.gdf_wm_as_pixc.to_file(str_fpath_wm_out_pixc_shp)

            return str_fpath_wm_out_pixc_shp

        elif fmt == "shp":

            str_fpath_wm_out_pixc_shp = os.path.join(str_fpath_dir_out, str_basename + ".shp")

            gdf_polygons = self.get_polygons(bool_clean=bool_clean, bool_label=bool_label, bool_exterior_only=False)
            gdf_polygons["indices"] = gdf_polygons["indices"].apply(str)

            gdf_polygons.to_file(str_fpath_wm_out_pixc_shp)
            return str_fpath_wm_out_pixc_shp

        else:
            raise ValueError("Unknown expected output format for the watermask")
