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

import logging

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import MultiPolygon, shape

_logger = logging.getLogger("widths_module")


def compute_widths_from_single_watermask(
    scenario, watermask, sections, buffer_length=25.0, index_attr="index", **kwargs
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

    if scenario == 0:  # Estimate widths only without uncertainty estimation
        return compute_widths_from_single_watermask_base(
            watermask,
            sections,
            buffer_length=buffer_length,
            index_attr=index_attr,
            **kwargs
        )

    elif scenario == 1:  # Estimate widths + uncertainty from section intersections
        raise NotImplementedError("Scenario 1 not implemented yet..")
    elif scenario == 10:  # Estimate widths + count banks within buffer
        raise NotImplementedError("Scenario 10 not implemented yet..")
    elif scenario == 11:  # scenario 1 + scenario 10
        return compute_widths_from_single_watermask_scenario11(
            watermask,
            sections,
            buffer_length=buffer_length,
            index_attr=index_attr,
            **kwargs
        )
    else:
        raise ValueError("Undefined scenario value")


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
            fname_buffered_section : str
                Path to the output file that will contain the buffered cross-sections. Default is None (no output).
    :return updated_sections: geopandas.GeoDataFrame
        For each input sections, enriched dataset with the width,
        the area of the buffer drawn around the section,
        and a boolean flag indicating if the buffer is full of water
    :return sections_buffered: gpd.GeoSeries
        If export_buffered_section is activated, return the bufferized section geometries
    """

    # Parse extra argument keywords
    label_attr = None
    if "label_attr" in kwargs:
        label_attr = kwargs["label_attr"]
    bool_print_dry = False
    if "bool_print_dry" in kwargs:
        bool_print_dry = kwargs["bool_print_dry"]
    min_width = None
    if "min_width" in kwargs:
        min_width = kwargs["min_width"]
    export_buffered_sections = False
    if "export_buffered_sections" in kwargs:
        export_buffered_sections = kwargs["export_buffered_sections"]

    # Check classes of input parameters
    if not isinstance(sections, gpd.GeoDataFrame):
        raise ValueError("sections must be a geopandas GeoDataFrame")
    if not isinstance(watermask, rio.DatasetReader):
        raise ValueError("watermask must be a rasterio DatasetReader")

    if min_width is not None:
        if not isinstance(min_width, int) and not isinstance(min_width, float):
            raise ValueError("min_width must be a number")

    if not isinstance(export_buffered_sections, bool):
        raise ValueError("export_buffered_sections must be True or False")

    # Create updated_sections GeoDataFrame (result of this function)
    updated_sections = sections.copy()
    # Add a column to contain estimated width
    updated_sections.insert(len(updated_sections.columns) - 1, "width", np.nan)
    # Add a column to contain buffer area
    updated_sections.insert(len(updated_sections.columns) - 1, "buffarea", np.nan)
    # Add a column to contain flag indicating if buffer is full of water (and so potentially sections is too short)
    updated_sections.insert(len(updated_sections.columns) - 1, "flg_bufful", np.nan)

    # Project sections to EPSG 3857 if necessary (to get metric distances)
    if sections.crs.to_epsg() == 4326:
        _logger.warning(
            "Inputs in epsg:4326 are projected to epsg:3857, not effective away from equator."
        )
        sections = sections.to_crs(epsg=3857)

    # Apply buffer to sections
    sections_buffered = sections.buffer(0.5 * buffer_length, cap_style=2)
    updated_sections["buffarea"] = sections_buffered.area

    # Export
    if export_buffered_sections:
        sections_buffered.to_file("sections_buffered.shp")

    # Compute pixel area
    pixel_area = watermask.transform[0] * np.abs(watermask.transform[4])

    for section_index in sections.index:

        # Mask the water mask with the buffer of current section
        section_buffered = sections_buffered.loc[section_index]
        if section_buffered.is_empty:
            updated_sections.loc[section_index, "width"] = np.nan

        else:
            try:
                out_image, out_transform = mask(
                    watermask,
                    shapes=[section_buffered],
                    crop=True,
                    nodata=watermask.nodata,
                )

                # Count number of water cells
                if label_attr is None:
                    water_pixels = np.sum(out_image != watermask.nodata)
                else:
                    int_label = sections.at[section_index, label_attr]
                    water_pixels = np.sum(out_image == int_label)
                water_area = water_pixels * pixel_area

                # Compute widths from area / buffer_length
                effective_width = water_area / buffer_length
                if min_width is not None:
                    if effective_width < min_width:
                        effective_width = min_width

                # Update width value in the GeoDataFrame
                updated_sections.at[section_index, "width"] = effective_width

                # Check if buffer is full
                if updated_sections.at[section_index, "buffarea"] == water_area:
                    updated_sections.loc[section_index, "flg_bufful"] = 1
                else:
                    updated_sections.loc[section_index, "flg_bufful"] = 0

            except Exception:
                updated_sections.loc[section_index, "width"] = np.nan

    # Print the dry sections
    if bool_print_dry:
        dry_sections = updated_sections[updated_sections["width"] < 1e-6]
        for section_index in range(dry_sections.shape[0]):
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

    # Parse extra argument keywords
    label_attr = None
    if "label_attr" in kwargs:
        label_attr = kwargs["label_attr"]
    bool_print_dry = False
    if "bool_print_dry" in kwargs:
        bool_print_dry = kwargs["bool_print_dry"]
    min_width = None
    if "min_width" in kwargs:
        min_width = kwargs["min_width"]
    export_buffered_sections = False
    if "export_buffered_sections" in kwargs:
        export_buffered_sections = kwargs["export_buffered_sections"]

    # Check classes of input parameters
    if not isinstance(sections, gpd.GeoDataFrame):
        raise ValueError("sections must be a geopandas GeoDataFrame")
    if not isinstance(watermask, rio.DatasetReader):
        raise ValueError("watermask must be a rasterio DatasetReader")

    if min_width is not None:
        if not isinstance(min_width, int) and not isinstance(min_width, float):
            raise ValueError("min_width must be a number")

    if not isinstance(export_buffered_sections, bool):
        raise ValueError("export_buffered_sections must be True or False")

    # Create updated_sections GeoDataFrame (result of this function)
    updated_sections = sections.copy()
    # Add a column to contain estimated width
    updated_sections.insert(len(updated_sections.columns) - 1, "width", np.nan)
    # Add a column to contain buffer area
    updated_sections.insert(len(updated_sections.columns) - 1, "buffarea", np.nan)
    # Add a flag column indicating if buffer is full of water (and so potentially sections is too short)
    updated_sections.insert(len(updated_sections.columns) - 1, "flg_bufful", np.nan)
    # Add a column to contain fraction of buffer area intersecting other buffers
    updated_sections.insert(len(updated_sections.columns) - 1, "beta", np.nan)
    # Add a column to contain number of river banks in buffer
    updated_sections.insert(len(updated_sections.columns) - 1, "nb_banks", np.nan)

    # Project sections to EPSG 3857 if necessary (to get metric distances)
    if sections.crs.to_epsg() == 4326:
        _logger.warning(
            "Inputs in epsg:4326 are projected to epsg:3857, not effective away from equator."
        )
        sections = sections.to_crs(epsg=3857)

    # Apply buffer to sections and store their area
    sections_buffered = sections.buffer(0.5 * buffer_length, cap_style=2)
    updated_sections["buffarea"] = sections_buffered.area

    # Export
    if export_buffered_sections:
        sections_buffered.to_file("sections_buffered.shp")

    # Compute pixel area
    pixel_area = watermask.transform[0] * np.abs(watermask.transform[4])

    l_shape = []
    l_buffer_waterarea = []
    l_nb_banks = []
    for section_index in sections.index:

        # Mask the water mask with the buffer of current section
        section_buffered = sections_buffered.loc[section_index]

        if section_buffered.is_empty or section_buffered is None:
            updated_sections.loc[section_index, "width"] = np.nan
            l_shape.append(MultiPolygon())
            l_buffer_waterarea.append(0.0)
            l_nb_banks.append(0.0)

        else:
            try:
                out_image, out_transform = mask(
                    watermask,
                    shapes=[section_buffered],
                    crop=True,
                    nodata=watermask.nodata,
                )

                # Count number of water cells
                if label_attr is None:
                    water_pixels = np.sum(out_image != watermask.nodata)
                else:
                    int_label = sections.at[section_index, label_attr]
                    water_pixels = np.sum(out_image == int_label)
                water_area = water_pixels * pixel_area

                # Compute widths from area / buffer_length
                effective_width = water_area / buffer_length
                if min_width is not None:
                    if effective_width < min_width:
                        effective_width = min_width

                # Update width value in the GeoDataFrame
                updated_sections.at[section_index, "width"] = effective_width

                # Check if buffer is full
                if updated_sections.at[section_index, "buffarea"] == water_area:
                    updated_sections.loc[section_index, "flg_bufful"] = 1
                else:
                    updated_sections.loc[section_index, "flg_bufful"] = 0

                # Compute water buffer polygon
                if label_attr is None:
                    l_geom_water_pols = [
                        shape(feat)
                        for feat, value in shapes(
                            out_image,
                            mask=(out_image != watermask.nodata),
                            transform=out_transform,
                        )
                    ]
                else:
                    int_label = sections.at[section_index, label_attr]
                    l_geom_water_pols = [
                        shape(feat)
                        for feat, value in shapes(
                            out_image,
                            mask=(out_image == int_label),
                            transform=out_transform,
                        )
                    ]

                l_shape.append(MultiPolygon(l_geom_water_pols))
                l_buffer_waterarea.append(water_area)
                l_nb_banks.append(2 * len(l_geom_water_pols))

            except Exception:
                updated_sections.loc[section_index, "width"] = np.nan
                l_shape.append(MultiPolygon())
                l_buffer_waterarea.append(0.0)
                l_nb_banks.append(0.0)

    # Gather waterbody details for each buffer
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

    # Quantify intersection between buffer shapes
    for index in gdf_waterbuffer.index:

        if not gdf_waterbuffer.at[index, "geometry"].is_empty:

            # Extract current buffer to check
            geom = gdf_waterbuffer.at[index, "geometry"].buffer(0)

            # Keep all other buffers apart
            gser_wrk = gdf_waterbuffer["geometry"].buffer(0).copy(deep=True)
            gser_wrk.drop(labels=index, inplace=True)
            gser_wrk = gser_wrk[~gser_wrk.is_empty]

            ser_buffer_intersection = gser_wrk.intersection(geom)

            ser_buffer_intersection_areatot = ser_buffer_intersection.area
            gdf_waterbuffer.at[index, "intersect_area"] = (
                ser_buffer_intersection_areatot.sum()
            )

            if gdf_waterbuffer.at[index, "water_area"] != 0.0:
                beta = (
                    gdf_waterbuffer.at[index, "intersect_area"]
                    / gdf_waterbuffer.at[index, "water_area"]
                )
            else:
                beta = np.nan

            if np.isnan(beta) or np.isinf(beta):
                beta = 0.0
            gdf_waterbuffer.at[index, "beta"] = beta

            del gser_wrk

    gdf_waterbuffer["flg_bufful"] = updated_sections["flg_bufful"]
    updated_sections["beta"] = gdf_waterbuffer["beta"]

    # Get waterbody banks number
    updated_sections["nb_banks"] = gdf_waterbuffer["nb_banks"]

    # Print the dry sections
    if bool_print_dry:
        dry_sections = updated_sections[updated_sections["width"] < 1e-6]
        for section_index in range(dry_sections.shape[0]):
            dry_section = dry_sections.iloc[section_index, :]
            _logger.info(
                "Dry section: %i (ID=%s)" % (section_index, dry_section[index_attr])
            )

    return updated_sections, sections_buffered
