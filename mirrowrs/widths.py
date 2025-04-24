# Copyright (C) 2024-2025 CS GROUP, https://csgroup.eu
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
widths.py
: contains functions to compute widths from a list of cross-sections over a watermask
"""

from dataclasses import dataclass
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import Polygon, MultiPolygon, shape

_logger = logging.getLogger("widths_module")

@dataclass
class ParamWidthComp:
    """kwargs parameters to compute width
    """
    label_attr: str = ""
    bool_print_dry: bool = False
    min_width: float = -1.
    export_buffered_sections: bool = False
    fname_buffered_section: str = "sections_buffered.shp"

    def __post_init__(self):
        """Check if attributes have the right class
        """
        if not isinstance(self.label_attr, str):
            raise TypeError("label_attr must be a str")

        if not isinstance(self.export_buffered_sections, bool):
            raise TypeError("export_buffered_sections must be a str")

        if not isinstance(self.min_width, (int, float)):
            raise TypeError("min_width must be a number")

        if not isinstance(self.bool_print_dry, bool):
            raise TypeError("export_buffered_sections must be True or False")

        if not isinstance(self.fname_buffered_section, str):
            raise TypeError("fname_buffered_section must be a str")

        if not self.fname_buffered_section[-4:]==".shp":
            raise ValueError("String must be a shapefile pathname.")


def count_pixels(out_image=None, val_nodata=255, bool_label=False, int_label=None):
    """Count water pixels in input band

    :param out_image: np.ndarray
        Watermask band
    :param val_nodata: int or float
        Value to exclude from count
    :param bool_label: bool
        If True, keep only label value in count
    :param int_label: int
        Label value to include in count
    :return water_pixels: int
        Count of water pixels
    """

    # Set _logger
    _logger = logging.getLogger("widths_module.count_pixels")

    # Check inputs
    if out_image is None:
        raise ValueError("Missing out_image input.")
    if not isinstance(out_image, np.ndarray):
        raise TypeError("Wrong type for input out_image")
    if not isinstance(val_nodata, (int, float)):
        raise TypeError("Input val_nodata must be numeric.")
    if not isinstance(bool_label, bool):
        raise TypeError("Input bool_label must be a boolean.")
    if bool_label:
        if int_label is None:
            raise ValueError("As bool_label is True, input int_label is missing.")
        if not isinstance(int_label, int):
            raise ValueError("Input int_label must be an integer.")

    if not bool_label:
        if np.isnan(val_nodata):
            water_pixels = np.sum(~np.isnan(out_image))
        elif np.isinf(val_nodata):
            water_pixels = np.sum(~np.isinf(out_image))
        else:
            water_pixels = np.sum(out_image != val_nodata)
    else:
        water_pixels = np.sum(out_image == int_label)

    return water_pixels

def compute_width_over_one_section(pol_section_buffered=None,
                                   flt_buffer_length=25.,
                                   flt_buffer_area=None,
                                   watermask=None,
                                   config=None,
                                   pixel_area=None,
                                   int_label=None):
    """Compute the effective width over a single buffered cross-section

    :param pol_section_buffered: Polygon or None
        Buffered cross-section
    :param flt_buffer_length: float
        Length of the section buffer
    :param flt_buffer_area: float
        Area of the buffered polygon
    :param watermask: rasterio.DatasetReader
            Raster of the water mask (any positive value is considered as water).
    :param config: ParamWidthComp
        Specific additionnal configurarion parameters for width computation
    :param pixel_area: float
        Area of a watermask pixel
    :param int_label: int
        Watermask label over width compute widths, if None, the full watermask is considered
    :return flt_effective_width: float
        Effective width
    :return flg_bufful: int
        1 if the buffer is fully covered of water, else 0
    :return out_image: np.ndarray
        cropped watermask within buffer
    :return out_transform: transform
        transformation of cropped watermask within buffer
    """

    # Set _logger
    _logger = logging.getLogger("widths_module.compute_width_over_one_section")

    # Check inputs
    if pol_section_buffered is not None:
        if not isinstance(pol_section_buffered, Polygon):
            raise TypeError("Wrong input type pol_section_buffered")
    for in_param, in_name in zip([flt_buffer_area, watermask, config, pixel_area], ["flt_buffer_area", "watermask", "config", "pixel_area"]):
        if in_param is None:
            raise ValueError(f"Missing input {in_name}")

    try:
        # Mask the water mask with the buffer of current section
        out_image, out_transform = mask(
                watermask,
                shapes=[pol_section_buffered],
                crop=True,
                nodata=watermask.nodata,
        )

        # Count number of water cells and compute water area
        water_pixels = count_pixels(out_image=out_image,
                                          val_nodata=watermask.nodata,
                                          bool_label=(config.label_attr != ""),
                                          int_label=int_label)

        # Compute widths from area / buffer_length
        water_area = water_pixels * pixel_area
        flt_effective_width = water_area / flt_buffer_length
        if config.min_width > 0.:
            if flt_effective_width < config.min_width:
                flt_effective_width = config.min_width

        # Check if buffer is full
        flg_bufful = 0
        if flt_buffer_area == water_area:
            flg_bufful = 1

    # Includes: pol_section_buffered.is_empty, pol_section_buffered is None
    except Exception as err:
        _logger.error(f"Error was raised: {err}")
        flt_effective_width = np.nan
        flg_bufful = 0
        out_image = None
        out_transform = None

    return flt_effective_width, flg_bufful, out_image, out_transform

def quantify_intersection_ratio_between_buffer(gdf_waterbuffer):
    """For each buffered section, compute the ratio of its area that intersects other buffers

    :param gdf_waterbuffer: gpd.GeoDataFrame
        Buffered geometries information: geometry and water_area ie the area of water within the buffer
    :return ser_beta: pd.Series
        A series with the same index as input gdf_waterbuffer
        containing the parameter beta ie ratio of buffer area that intersects other buffers
    """

    # Set _logger
    _logger = logging.getLogger("widths_module.quantify_intersection_ratio_between_buffer")

    # Initiate beta parameter
    ser_beta = pd.Series(index=gdf_waterbuffer.index)
    ser_beta[ser_beta.isna()] = 0.

    # Check each section
    for index in gdf_waterbuffer.index:

        # Check if section buffer is not empty
        if not gdf_waterbuffer.at[index, "geometry"].is_empty:

            # Extract current buffer to check
            geom = gdf_waterbuffer.at[index, "geometry"].buffer(0)

            # Keep all other buffers apart
            gser_wrk = gdf_waterbuffer["geometry"].buffer(0).copy(deep=True)
            gser_wrk.drop(labels=index, inplace=True)
            gser_wrk = gser_wrk[~gser_wrk.is_empty]

            # Estimate intersection between current buffer and all others
            ser_buffer_intersection = gser_wrk.intersection(geom)

            # Compute area of each intersection
            ser_buffer_intersection_areatot = ser_buffer_intersection.area

            # Sum up all intersection areas
            gdf_waterbuffer.at[index, "intersect_area"] = (
                ser_buffer_intersection_areatot.sum()
            )

            flt_beta = 0.
            if gdf_waterbuffer.at[index, "water_area"] != 0.0:
                flt_beta = (
                    gdf_waterbuffer.at[index, "intersect_area"]
                    / gdf_waterbuffer.at[index, "water_area"]
                )

            ser_beta.loc[index] = flt_beta

            # Clean variable
            del gser_wrk

    # gdf_waterbuffer["flg_bufful"] = updated_sections["flg_bufful"]
    return ser_beta


def compute_widths_from_single_watermask_base(
    watermask, sections, buffer_length=25.0, index_attr="index", **kwargs
):
    """Compute the widths - and only width - for a list of cross-sections using a watermask

    :param watermask: rasterio.DatasetReader
        Raster of the water mask (any positive value is considered as water).
    :param sections: geopandas.GeoDataFrame
        GeoDataFrame containing (at least) the cross-sections geometry and the length and index attributes.
    :param buffer_length: float
        Length of the buffer to apply to cross-sections.
    :param index_attr: str
        Name of the attribute that contains indexes in the GeoDataFrame.
    :param kwargs: dict
        Other keyword arguments:
            label_attr : str
                in sections argument, attribute name containing segmentation label for each section
            bool_print_dry : bool
                If true, print out information on possible dry sections
            min_width : float
                Value of the minimal width.
            export_buffered_sections : bool
                If True, save buffered section geometry
            fname_buffered_section : str
                Path to the output file that will contain the buffered cross-sections. Default is None (no output).
    :return updated_sections: geopandas.GeoDataFrame
        For each input sections, enriched dataset with the width,
        the area of the buffer drawn around the section,
        and a boolean flag indicating if the buffer is full of water
    :return sections_buffered: gpd.GeoSeries
        If export_buffered_section is activated, return the bufferized section geometries
    """

    # Set _logger
    _logger = logging.getLogger("widths_module.compute_widths_from_single_watermask_base")

    # Parse extra argument keywords
    config = ParamWidthComp(**kwargs)

    # Check input parameters
    if not isinstance(sections, gpd.GeoDataFrame):
        raise TypeError("sections must be a geopandas GeoDataFrame")
    if not isinstance(watermask, rio.DatasetReader):
        raise TypeError("watermask must be a rasterio DatasetReader")

    # Project sections to EPSG 3857 if necessary (to get metric distances)
    if sections.crs.to_epsg() == 4326:
        _logger.warning(
            "Inputs in epsg:4326 are projected to epsg:3857, not effective away from equator."
        )
        sections = sections.to_crs(epsg=3857)

    # Create updated_sections GeoDataFrame (result of this function)
    updated_sections = sections.copy()

    # Add attributes dedicated to width
    l_new_attr = ["width", "buffarea", "flg_bufful"]
    for attr in l_new_attr:
        updated_sections.insert(len(updated_sections.columns) - 1, attr, np.nan)

    # Apply buffer to sections
    sections_buffered = sections.buffer(0.5 * buffer_length, cap_style="flat")
    updated_sections["buffarea"] = sections_buffered.area

    # Export
    if config.export_buffered_sections:
        sections_buffered.to_file(config.fname_buffered_section)

    # Compute pixel area
    pixel_area = watermask.transform[0] * np.abs(watermask.transform[4])

    for section_index in sections.index:

        # Get buffered section geometry
        section_buffered = sections_buffered.loc[section_index]

        int_label = None
        if config.label_attr != "":
            int_label = int(sections.at[section_index, config.label_attr])

        # Compute effective width at section
        flt_effective_width, flg_bufful, _, _ = compute_width_over_one_section(pol_section_buffered=section_buffered,
                                                                         flt_buffer_length=buffer_length,
                                                                         flt_buffer_area=updated_sections.loc[section_index,"buffarea"],
                                                                         watermask=watermask,
                                                                         config=config,
                                                                         pixel_area=pixel_area,
                                                                         int_label=int_label)

        # Update parameters
        updated_sections.loc[section_index, "width"] = flt_effective_width
        updated_sections.loc[section_index, "flg_bufful"] = flg_bufful

    # Print the dry sections
    if config.bool_print_dry:
        # dry_sections = updated_sections[updated_sections["width"].isna()]
        dry_sections = updated_sections[updated_sections["width"] < 1.e-6]
        for section_index in dry_sections.index:
            dry_section = dry_sections.iloc[section_index, :]
            _logger.info(
                "Dry section: %i (ID=%s)" % (section_index, dry_section[index_attr])
            )

    return updated_sections, sections_buffered


def compute_widths_from_single_watermask_scenario11(
    watermask, sections, buffer_length=25.0, index_attr="index", **kwargs
):
    """Compute the widths for a list of cross-sections using a watermask

    :param watermask: rasterio.DatasetReader
        Raster of the water mask (any positive value is considered as water).
    :param sections: geopandas.GeoDataFrame
        GeoDataFrame containing (at least) the cross-sections geometry and the length and index attributes.
    :param buffer_length: float
        Length of the buffer to apply to cross-sections.
    :param index_attr: str
        Name of the attribute that contains indexes in the GeoDataFrame.
    :param kwargs: dict
        Other keyword arguments:
            label_attr : str
                in sections argument, attribute name containing segmentation label for each section
            bool_print_dry : bool
                If true, print out information on possible dry sections
            min_width : float
                Value of the minimal width.
            export_buffered_sections : bool
                If True, save buffered section geometry
            fname_buffered_section : str
                Path to the output file that will contain the buffered cross-sections. Default is None (no output).
    :return updated_sections: geopandas.GeoDataFrame
        For each input sections, enriched dataset with the width,
        the area of the buffer drawn around the section,
        a boolean flag indicating if the buffer is full of water,
        a parameter measuring the portion of buffer area intersecting neighboring buffer,
        a count of the number of banks in the buffer
    :return sections_buffered: gpd.GeoSeries
        If export_buffered_section is activated, return the bufferized section geometries
    """

    # Set _logger
    _logger = logging.getLogger("widths_module.compute_widths_from_single_watermask_scenario11")

    # Parse extra argument keywords
    config = ParamWidthComp(**kwargs)

    # Check classes of input parameters
    if not isinstance(sections, gpd.GeoDataFrame):
        raise TypeError("sections must be a geopandas GeoDataFrame")
    if not isinstance(watermask, rio.DatasetReader):
        raise TypeError("watermask must be a rasterio DatasetReader")

    # Project sections to EPSG 3857 if necessary (to get metric distances)
    if sections.crs.to_epsg() == 4326:
        _logger.warning(
            "Inputs in epsg:4326 are projected to epsg:3857, not effective away from equator."
        )
        sections = sections.to_crs(epsg=3857)

    # Create updated_sections GeoDataFrame (result of this function)
    updated_sections = sections.copy()

    # Add attributes dedicated to width
    l_new_attr = ["width", "buffarea", "flg_bufful", "beta", "nb_banks"]
    for attr in l_new_attr:
        updated_sections.insert(len(updated_sections.columns) - 1, attr, np.nan)

    # Apply buffer to sections and store their area
    sections_buffered = sections.buffer(0.5 * buffer_length, cap_style=2)
    updated_sections["buffarea"] = sections_buffered.area

    # Export
    if config.export_buffered_sections:
        sections_buffered.to_file("sections_buffered.shp")

    # Compute pixel area
    pixel_area = watermask.transform[0] * np.abs(watermask.transform[4])

    l_shape = []
    l_buffer_waterarea = []
    l_nb_banks = []
    for section_index in sections.index:

        # Get buffered section geometry
        section_buffered = sections_buffered.loc[section_index]

        int_label = None
        if config.label_attr != "":
            int_label = int(sections.at[section_index, config.label_attr])

        # Compute effective width at section
        flt_effective_width, flg_bufful, out_image, out_transform = compute_width_over_one_section(pol_section_buffered=section_buffered,
                                                                         flt_buffer_length=buffer_length,
                                                                         flt_buffer_area=updated_sections.loc[
                                                                             section_index, "buffarea"],
                                                                         watermask=watermask,
                                                                         config=config,
                                                                         pixel_area=pixel_area,
                                                                         int_label=int_label)

        # Update parameters
        updated_sections.loc[section_index, "width"] = flt_effective_width
        updated_sections.loc[section_index, "flg_bufful"] = flg_bufful

        if np.isnan(flt_effective_width):
            l_shape.append(MultiPolygon())
            l_buffer_waterarea.append(0.0)
            l_nb_banks.append(0.0)

        else:
            if config.label_attr == "":
                l_geom_water_pols = [
                    shape(feat)
                    for feat, value in shapes(
                        out_image,
                        mask=(out_image != watermask.nodata),
                        transform=out_transform,
                    )
                ]
            else:
                int_label = int(sections.at[section_index, config.label_attr])
                l_geom_water_pols = [
                    shape(feat)
                    for feat, value in shapes(
                        out_image,
                        mask=(out_image == int_label),
                        transform=out_transform,
                    )
                ]

            l_shape.append(MultiPolygon(l_geom_water_pols))
            l_buffer_waterarea.append(flt_effective_width*buffer_length)
            l_nb_banks.append(2 * len(l_geom_water_pols))

    # Gather waterbody details from each buffer
    gdf_waterbuffer = gpd.GeoDataFrame(
        {
            "water_area": l_buffer_waterarea,
            "nb_banks": l_nb_banks,
            "intersect_area": 0.0,
            "beta": 0.0,
        },
        index=sections.index,
        geometry=gpd.GeoSeries(l_shape, crs=sections.crs, index=sections.index),
        crs=sections.crs,
    )
    updated_sections["nb_banks"] = gdf_waterbuffer["nb_banks"]

    # Retrieve ratio of area intersection
    ser_beta = quantify_intersection_ratio_between_buffer(gdf_waterbuffer)
    updated_sections["beta"] = ser_beta

    # Print the dry sections
    if config.bool_print_dry:
        dry_sections = updated_sections[updated_sections["width"] < 1e-6]
        for section_index in range(dry_sections.shape[0]):
            dry_section = dry_sections.iloc[section_index, :]
            _logger.info(
                "Dry section: %i (ID=%s)" % (section_index, dry_section[index_attr])
            )

    return updated_sections, sections_buffered

def compute_widths_from_single_watermask(
        scenario,
        watermask,
        sections,
        buffer_length=25.0,
        index_attr="index",
        **kwargs
):
    """Compute the widths for a list of cross-sections using a single watermask

    :param scenario: int
        Width computation scenario
    :param watermask: rasterio.DatasetReader
            Raster of the water mask (any positive value is considered as water).
    :param sections: geopandas.GeoDataFrame
            GeoDataFrame containing (at least) the cross-sections geometry and the length and index attributes.
    :param buffer_length: float
            Length of the buffer to apply to cross-sections.
    :param index_attr: str
            Name of the attribute that contains indexes in the GeoDataFrame.
    :param kwargs: dict
            Other keyword arguments:
            label_attr : str
                in sections argument, attribute name containing segmentation label for each section
            bool_print_dry : bool
                If true, print out information on possible dry sections
            min_width : float
                Value of the minimal width.
            fname_buffered_section : str
                Path to the output file that will contain the buffered cross-sections. Default is None (no output).

    :return updated_sections: geopandas.GeoDataFrame
        For each input sections, enriched dataset with the width,
        the area of the buffer drawn around the section,
        and a boolean flag indicating if the buffer is full of water
    :return sections_buffered: gpd.GeoSeries
        If export_buffered_section is activated, return the bufferized section geometries
    """

    # Set _logger
    _logger = logging.getLogger("widths_module.compute_widths_from_single_watermask")

    # Check scenario values
    if scenario not in [0, 1, 10, 11]:
        raise ValueError("Non-value scenario value")

    if scenario in [1, 10]:
        raise NotImplementedError(f"Scenario {scenario} not implemented yet..")
        # 1 :: Estimate widths + uncertainty from section intersections
        #10 :: # Estimate widths + count banks within buffer

    # Run the right function given the scenario
    if scenario == 0:  # Estimate widths only without uncertainty estimation
        updated_sections, sections_buffered = compute_widths_from_single_watermask_base(
            watermask,
            sections,
            buffer_length=buffer_length,
            index_attr=index_attr,
            **kwargs
        )

    if scenario == 11:  # scenario 1 + scenario 10
        updated_sections, sections_buffered =  compute_widths_from_single_watermask_scenario11(
            watermask,
            sections,
            buffer_length=buffer_length,
            index_attr=index_attr,
            **kwargs
        )

    return updated_sections, sections_buffered