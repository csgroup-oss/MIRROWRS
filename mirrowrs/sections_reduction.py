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
sections_reduction.py
: tools to reduce the geometry of cross-sections
"""

import logging

import numpy as np
from shapely.geometry import (GeometryCollection, LineString, MultiLineString,
                              MultiPoint, Point, Polygon, MultiPolygon)

from mirrowrs.constants import FLT_TOL_DIST_DEFAULT, FLT_TOL_LEN_DEFAULT

_logger = logging.getLogger("reduce_sections_module")


def link_multilinestring_pieces(multiline_in, l_idx=None):
    """Turn a multilinestring object into a linestring

    :param multiline_in: MultiLineString
        multilinestring geometry object to merge
    :param l_idx: list
        sorting of multilinestring elements
    :return lin_out: LineString
    """

    l_xy = []
    if l_idx is None:
        for geom in multiline_in.geoms:
            l_xy += list(geom.coords)
    else:

        npar_idx_sort = np.argsort(np.array(l_idx))

        for idx in npar_idx_sort:
            l_xy += list(list(multiline_in.geoms)[idx].coords)
    lin_out = LineString(l_xy)

    return lin_out


def reduce_section(lin_long_in, pol_in, how="simple", **kwargs):
    """Reduce linestring to the shortest linestring with the control polygon

    :param lin_long_in: LineString
        Input long line geometry to reduce
    :param pol_in: Polygon
        Close domain within which reduce the input line
    :param how: str
        algorithm/method on how to reduce : 'simple', 'hydrogeom'
    :param kwargs:
        'lin_rch_in': (MultiLineString, LineString)
            geometry of centerline
        'flt_section_xs_along_rch': float
            curvilinear abscissa of node associated with the current section along the centerline
        'int_nb_chan_max': int
            maximum number of possible channels covered by the cross-section, default 1
        'flt_node_proj_x': float
            (in the wm crs), x-coordinate of the node associated with the current section
        'flt_node_proj_y': float
            (in the wm crs), y-coordinate of the node associated with the current section
        'flt_tol_len': float
            in ]0., 1.] minimum length ratio of sub-section compared to total length to keep the sub-section, default 0.05
        'flt_tol_dist': float
            maximum distance [m] to node of the sub-section mid-point projected on the centerline, default 400.

    :return lin_out: LineString
        Reduced input line
    """

    # Set _logger
    _logger = logging.getLogger("reduce_sections_module.reduce_section")

    if how == "simple":
        lin_out = reduce_section_simple(lin_long_in, pol_in)

    elif how == "hydrogeom":

        if "lin_rch_in" in kwargs:
            in_lin_rch_in = kwargs["lin_rch_in"]
        else:
            in_lin_rch_in = None

        if "flt_section_xs_along_rch" in kwargs:
            in_flt_section_xs_along_rch = kwargs["flt_section_xs_along_rch"]
        else:
            in_flt_section_xs_along_rch = None

        if "int_nb_chan_max" in kwargs:
            in_int_nb_chan_max = kwargs["int_nb_chan_max"]
        else:
            in_int_nb_chan_max = 1

        if "flt_node_proj_x" in kwargs:
            in_flt_node_proj_x = kwargs["flt_node_proj_x"]
        else:
            in_flt_node_proj_x = None

        if "flt_node_proj_y" in kwargs:
            in_flt_node_proj_y = kwargs["flt_node_proj_y"]
        else:
            in_flt_node_proj_y = None

        if "flt_tol_len" in kwargs:
            in_flt_tol_len = kwargs["flt_tol_len"]
        else:
            in_flt_tol_len = FLT_TOL_LEN_DEFAULT

        if "flt_tol_dist" in kwargs:
            in_flt_tol_dist = kwargs["flt_tol_dist"]
        else:
            in_flt_tol_dist = FLT_TOL_DIST_DEFAULT

        lin_out = reduce_section_hydrogeom(
            lin_long_in,
            pol_in,
            lin_rch_in=in_lin_rch_in,
            flt_section_xs_along_rch=in_flt_section_xs_along_rch,
            int_nb_chan_max=in_int_nb_chan_max,
            flt_node_proj_x=in_flt_node_proj_x,
            flt_node_proj_y=in_flt_node_proj_y,
            flt_tol_len=in_flt_tol_len,
            flt_tol_dist=in_flt_tol_dist,
        )

    else:
        raise NotImplementedError(f"Input reduce method {how} does not exists")

    return lin_out


def reduce_section_simple(lin_long_in, pol_in):
    """Reduce linestring to the shortest linestring with the control polygon

    :param lin_long_in: LineString
        Input long line geometry to reduce
    :param pol_in: Polygon
        Close domain within which reduce the input line
    :return lin_out: LineString
        Reduced input line
    """

    # Set _logger
    _logger = logging.getLogger("reduce_sections_module.reduce_section_simple")

    # Check inputs
    if not isinstance(lin_long_in, (LineString, MultiLineString)):
        raise TypeError("Input lin_long_in must be a line-like object")
    if not isinstance(pol_in, (Polygon, MultiPolygon)):
        raise TypeError("Input pol_in must be a polygon-like object")

    lin_cut = pol_in.intersection(lin_long_in)

    if isinstance(lin_cut, MultiLineString):
        lin_out = link_multilinestring_pieces(lin_cut)
    elif isinstance(lin_cut, LineString):
        lin_out = lin_cut
    else:
        raise NotImplementedError("Can't handle (yet) case when reduced line is not a LineString/MultiLineString")

    return lin_out


def reduce_section_hydrogeom(
    lin_long_in,
    pol_in,
    lin_rch_in=None,
    flt_section_xs_along_rch=None,
    flt_node_proj_x=None,
    flt_node_proj_y=None,
    int_nb_chan_max=1,
    flt_tol_len=FLT_TOL_LEN_DEFAULT,
    flt_tol_dist=FLT_TOL_DIST_DEFAULT,
):
    """Reduce section following hydrological constraints

    :param lin_long_in: LineString
        geometry of long cross-section
    :param pol_in: (Polygon/MultiPolygon)
        polygon representing the shape of the watermask
    :param lin_rch_in: (MultiLineString, LineString)
        geometry of centerline
    :param flt_section_xs_along_rch: float
        curvilinear abscissa of node associated with the current section along the centerline
    :param flt_node_proj_x: float
        (in the wm crs), x-coordinate of the node associated with the current section
    :param flt_node_proj_y: float
        (in the wm crs), y-coordinate of the node associated with the current section
    :param int_nb_chan_max: int
        maximum number of possible channels covered by the cross-section, default 1
    :param flt_tol_len: float
        in ]0., 1.] minimum length ratio of sub-section compared to total length to keep the sub-section, default 0.05
    :param flt_tol_dist: float
        maximum distance [m] to node of the sub-section mid-point projected on the centerline, default 400.

    :return: (MultiLineString, LineString)
        lin_section_corrected : the reduced section with only valid parts, can be empty
    """

    # Set _logger
    _logger = logging.getLogger("reduce_sections_module.reduce_section_hydrogeom")

    # Check inputs
    if lin_long_in is None:
        raise ValueError("Missing input argument 'lin_long_in'")
    if not isinstance(lin_long_in, (LineString, MultiLineString)):
        raise TypeError("Input lin_long_in must be a line-like object")

    if pol_in is None:
        raise ValueError("Missing input argument 'pol_in'")
    if not isinstance(pol_in, (Polygon, MultiPolygon)):
        raise TypeError("Input pol_in must be a polygon-like object")

    if lin_rch_in is None:
        raise ValueError("Missing input argument 'lin_rch_in' as projected geometry of centerline")
    if not isinstance(lin_rch_in, LineString):
        raise TypeError("Input lin_rch_in must be a line-like object")

    if flt_section_xs_along_rch is None:
        raise ValueError(
            "Missing input argument 'flt_section_xs_along_rch' as curvilinear abscissa of node associated to the section along centerline"
        )
    if not isinstance(flt_section_xs_along_rch, float):
        raise TypeError("Input 'flt_section_xs_along_rch' must be numeric")

    if flt_node_proj_x is None:
        raise ValueError(
            "Missing input argument 'flt_node_proj_x' as x-coordinate of node in projected system"
        )
    if not isinstance(flt_node_proj_x, float):
        raise TypeError("Input 'flt_node_proj_x' must be numeric")

    if flt_node_proj_y is None:
        raise ValueError(
            "Missing input argument 'flt_node_proj_y' as y-coordinate of node in projected system"
        )
    if not isinstance(flt_node_proj_y, float):
        raise TypeError("Input 'flt_node_proj_y' must be numeric")

    # Intersection(s) between the long section and the wm
    geom_chn_cnt = pol_in.intersection(
        lin_long_in
    )

    # Intersection(s) between the long section and the centerline
    geom_rch_cnt = lin_long_in.intersection(
        lin_rch_in
    )

    # Clean intersections
    if isinstance(geom_chn_cnt, LineString):

        if isinstance(geom_rch_cnt, Point):
            lin_out = geom_chn_cnt

        else:
            lin_out = LineString()

    elif isinstance(geom_chn_cnt, MultiLineString):

        if isinstance(geom_rch_cnt, Point):
            lin_out = reduce_section_hydrogeom_multiline_point(
                lin_rch_in=lin_rch_in,
                lin_long_in=lin_long_in,
                flt_section_xs_along_rch=flt_section_xs_along_rch,
                geom_chn_cnt=geom_chn_cnt,
                int_nb_chan_max=int_nb_chan_max,
                node_proj_x=flt_node_proj_x,
                node_proj_y=flt_node_proj_y,
                flt_tol_len=flt_tol_len,
                flt_tol_dist=flt_tol_dist,
            )

        elif isinstance(geom_rch_cnt, MultiPoint):

            lin_out = reduce_section_hydrogeom_multiline_multipoint(
                lin_long_in=lin_long_in,
                geom_chn_cnt=geom_chn_cnt,
                geom_rch_cnt=geom_rch_cnt,
                lin_rch_in=lin_rch_in,
                flt_section_xs_along_rch=flt_section_xs_along_rch,
                node_proj_x=flt_node_proj_x,
                node_proj_y=flt_node_proj_y,
                int_nb_chan_max=int_nb_chan_max,
                flt_tol_len=flt_tol_len,
                flt_tol_dist=flt_tol_dist,
            )

        else:
            lin_out = LineString()

    elif isinstance(geom_chn_cnt, GeometryCollection):
        raise NotImplementedError(
            "Not implemented for GeometryCollection"
        )

    else:
        lin_out = LineString()

    return lin_out


def reduce_section_hydrogeom_multiline_point(
    lin_rch_in,
    lin_long_in,
    geom_chn_cnt,
    flt_section_xs_along_rch,
    node_proj_x,
    node_proj_y,
    int_nb_chan_max=1,
    flt_tol_len=FLT_TOL_LEN_DEFAULT,
    flt_tol_dist=FLT_TOL_DIST_DEFAULT,
):
    """Check each sub-section of the total raw cross-section that intersects the watermask
    and remove them from the reduced cross-section if not valid

    :param lin_rch_in: (MultiLineString, LineString)
        geometry of centerline
    :param lin_long_in: LineString
        geometry of long cross-section
    :param geom_chn_cnt: MultiLineString
        geometry of cross-section reduced to intersections with the watermask
    :param flt_section_xs_along_rch: float
        curvilinear abscissa of node associated with the current section along the centerline
    :param node_proj_x: float
        (in the wm crs), x-coordinate of the node associated with the current section
    :param node_proj_y: float
        (in the wm crs), y-coordinate of the node associated with the current section
    :param int_nb_chan_max: int
        maximum number of possible channels covered by the cross-section, default 1
    :param flt_tol_len: float
        in ]0., 1.] minimum length ratio of sub-section compared to total length to keep the sub-section, default 0.05
    :param flt_tol_dist: float
        maximum distance [m] to node of the sub-section mid-point projected on the centerline, default 400.

    :return: (MultiLineString, LineString)
        lin_section_corrected : the reduced section with only valid parts, can be empty

    """

    # Set _logger
    _logger = logging.getLogger("reduce_sections_module.reduce_section_hydrogeom_multiline_point")

    # Check input parameters
    if int_nb_chan_max <= 0:
        raise ValueError("'int_nb_chan_max' parameter must be strictly positive")
    if not isinstance(int_nb_chan_max, int):
        int_nb_chan_max = int(int_nb_chan_max)
        _logger.warning("Warning: 'int_nb_chan_max' has been converted into a int")
    if flt_tol_len > 1.0 or flt_tol_len <= 0.0:
        raise ValueError("'flt_tol_len' parameter must be in interval ]0.; 1.]")
    if flt_tol_dist < 0.0:
        raise ValueError("'flt_tol_dist' parameter must be positive")

    # xs along long-section of each sub-section mid-point
    npar_xs_subcross_cross = np.array(
        [
            lin_long_in.project(geom.interpolate(0.5, normalized=True))
            for geom in geom_chn_cnt.geoms
        ]
    )

    # to find in which sub-segment the node is
    flt_node_xs_along_xsec = lin_long_in.project(Point(node_proj_x, node_proj_y))

    # Sort channel given the distance to the centerline
    npar_idx_channel_dst2rch_argsrt = np.argsort(
        np.abs(npar_xs_subcross_cross - flt_node_xs_along_xsec)
    )

    # reproject subsection mid-point on centerline
    npar_xs_subcross_rch = np.array(
        [
            lin_rch_in.project(geom.interpolate(0.5, normalized=True))
            for geom in geom_chn_cnt.geoms
        ]
    )
    # to check if subsection is not too far away from the current node

    # Total cumulative length of sub-crossection
    flt_cumul_length = np.sum(np.array([geom.length for geom in geom_chn_cnt.geoms]))

    l_lin_section_corrected = []
    l_lin_section_idx = []
    for idx in npar_idx_channel_dst2rch_argsrt:

        # Criterion : nb channel
        bool_nb_channel = len(l_lin_section_corrected) <= int_nb_chan_max

        # Criterion : projection of subcrosssection on centerline not too fr from current node
        bool_dist2node = (
            abs(npar_xs_subcross_rch[idx] - flt_section_xs_along_rch) < flt_tol_dist
        )

        # Criterion : sub-crosssection not too small
        bool_len_subcross = (
            list(geom_chn_cnt.geoms)[idx].length / flt_cumul_length > flt_tol_len
        )

        if bool_nb_channel and bool_dist2node and bool_len_subcross:
            l_lin_section_corrected.append(list(geom_chn_cnt.geoms)[idx])
            l_lin_section_idx.append(idx)

    if len(l_lin_section_corrected) > 0:
        lin_section_corrected_base = MultiLineString(l_lin_section_corrected)
        lin_section_corrected = link_multilinestring_pieces(
            lin_section_corrected_base, l_lin_section_idx
        )
    else:
        lin_section_corrected = LineString()

    return lin_section_corrected


def reduce_section_hydrogeom_multiline_multipoint(
    lin_long_in,
    geom_chn_cnt,
    geom_rch_cnt,
    lin_rch_in,
    flt_section_xs_along_rch,
    node_proj_x,
    node_proj_y,
    int_nb_chan_max=1,
    flt_tol_len=FLT_TOL_LEN_DEFAULT,
    flt_tol_dist=FLT_TOL_DIST_DEFAULT,
):
    """Check each sub-section of the total raw cross-section that intersects the watermask
    and remove them from the reduced cross-section if not valid
    with additionnal criterion to check as the section intersects the centerline multiple times

    :param lin_rch_in: (MultiLineString, LineString)
        geometry of centerline
    :param lin_long_in: LineString
        geometry of long cross-section
    :param geom_chn_cnt: MultiLineString
        geometry of cross-section reduced to intersections with the watermask
    :param geom_rch_cnt : MultiPoint
        geometry of intersections between the centerline and the long cross-section
    :param flt_section_xs_along_rch: float
        curvilinear abscissa of node associated with the current section along the centerline
    :param node_proj_x: float
        (in the wm crs), x-coordinate of the node associated with the current section
    :param node_proj_y: float
        (in the wm crs), y-coordinate of the node associated with the current section
    :param int_nb_chan_max: int
        maximum number of possible channels covered by the cross-section, default 1
    :param flt_tol_len: float
        in ]0., 1.] minimum length ratio of sub-section compared to total length to keep the sub-section, default 0.05
    :param flt_tol_dist: float
        maximum distance [m] to node of the sub-section mid-point projected on the centerline, default 400.

    :return: (MultiLineString, LineString)
        lin_section_corrected : the reduced section with only valid parts, can be empty
    """

    # Set _logger
    _logger = logging.getLogger("reduce_sections_module.reduce_section_hydrogeom_multiline_multipoint")

    # Prepare criteria to check if subsection is valid to be kept
    # Total cumulative length of sub-crossection
    flt_cumul_length = np.sum(np.array([geom.length for geom in geom_chn_cnt.geoms]))

    # Reproject subsection mid-point on centerline
    # to later check if subsection is not too far away from the current node
    npar_xs_subcross_rch = np.array(
        [
            lin_rch_in.project(geom.interpolate(0.5, normalized=True))
            for geom in geom_chn_cnt.geoms
        ]
    )

    # Place intersectionS with centerline over its sub-sections
    # Among all intersection, idx of the one matching the current node
    npar_flt_subsection_inter_rch__xs_along_rch = np.array(
        [lin_rch_in.project(p) for p in geom_rch_cnt.geoms]
    )
    idx_cross_pt_node = np.argmin(
        np.abs(npar_flt_subsection_inter_rch__xs_along_rch - flt_section_xs_along_rch)
    )

    # Map rch and pt
    npar_int_map_cross = np.zeros(
        (len(list(geom_rch_cnt.geoms)), len(list(geom_chn_cnt.geoms))), dtype=np.uint8
    )
    for i, geom_pt in enumerate(geom_rch_cnt.geoms):
        for j, geom_rch in enumerate(geom_chn_cnt.geoms):
            if geom_pt.distance(geom_rch) < 10.0:
                npar_int_map_cross[i, j] = 1

    # idx of the rch containing the current node intersection (#column)
    idx_cross_rch_node = np.where(npar_int_map_cross[idx_cross_pt_node, :] == 1)[0]

    # xs along raw long section of each sub-section mid-point
    npar_flt_subsection_mid__xs_along_xsec = np.array(
        [
            lin_long_in.project(geom.interpolate(0.5, normalized=True))
            for geom in geom_chn_cnt.geoms
        ]
    )
    nb_pt_on_rch = np.sum(npar_int_map_cross[:, idx_cross_rch_node])
    # To be sorted in increasing order

    # node xs along long section
    flt_node_xs_along_xsec = lin_long_in.project(Point(node_proj_x, node_proj_y))
    npar_flt_subsection_inter_rch__xs_along_rch_ctr = (
        npar_flt_subsection_mid__xs_along_xsec - flt_node_xs_along_xsec
    )

    # Sort channel given the distance to the centerline
    npar_idx_channel_dst2rch_argsrt = np.argsort(
        np.abs(npar_flt_subsection_mid__xs_along_xsec - flt_node_xs_along_xsec)
    )

    if (
        nb_pt_on_rch > 1
    ):  # Non valid case where section crosses multiple times the centerline within a same channel
        lin_section_corrected = LineString()

    else:

        l_lin_section_corrected = []
        l_idx = []

        npar_bool_valid = np.zeros((len(list(geom_chn_cnt.geoms)),))
        int_nb_chan = 0
        # Loop over subsection from the closest to furthest to centerline
        for idx in npar_idx_channel_dst2rch_argsrt:

            geom_subsection = list(geom_chn_cnt.geoms)[idx]

            # Check if current subsection is valid
            if idx == npar_idx_channel_dst2rch_argsrt[0]:

                # The channel containing the centerline is by default valid
                # Non-valid case (multiple centerline intersection within the channel) has already been eliminated
                bool_valid = 1

            else:

                # Get number of xsec_inter_rch point on the current subsection
                cur_nb_pt_on_rch = np.sum(
                    npar_int_map_cross[:, idx]
                )  # subsection is valid if cur_nb_pt_on_rch < 1

                if (
                    cur_nb_pt_on_rch > 1
                ):  # The channel contains another part of the centerline : not valid
                    bool_valid = 0
                else:
                    # The channel does not contain another part of the centerline : could be valid
                    # valid flag = valid flag from adjacent channel (that has already been checked)
                    # Need debug/validation !!!
                    if (
                        npar_flt_subsection_inter_rch__xs_along_rch_ctr[idx] > 0
                    ):  # on a given side of the centerline

                        bool_valid = npar_bool_valid[idx - 1]
                    else:  # on the other side of the centerline
                        bool_valid = npar_bool_valid[idx + 1]

            # Additionnal criteria to add subsection
            # Criterion : nb channel
            bool_nb_channel = len(l_lin_section_corrected) <= int_nb_chan_max

            # Criterion : projection of subcrosssection on centerline not too far from current node
            bool_dist2node = (
                abs(npar_xs_subcross_rch[idx] - flt_section_xs_along_rch) < flt_tol_dist
            )

            # Criterion : sub-crosssection not too short
            bool_len_subcross = (
                list(geom_chn_cnt.geoms)[idx].length / flt_cumul_length > flt_tol_len
            )

            if bool_valid and bool_nb_channel and bool_dist2node and bool_len_subcross:
                l_lin_section_corrected.append(geom_subsection)
                l_idx.append(idx)
                int_nb_chan += 1

            else:
                npar_bool_valid[idx] = 0

        if len(l_lin_section_corrected) > 0:
            lin_section_corrected_base = MultiLineString(l_lin_section_corrected)
            lin_section_corrected = link_multilinestring_pieces(
                lin_section_corrected_base, l_idx
            )
        else:
            lin_section_corrected = LineString()

    return lin_section_corrected
