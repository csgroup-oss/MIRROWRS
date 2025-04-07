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
Functionnality tests
"""

import os

import geopandas as gpd
import rasterio as rio

from mirrowrs.basprocessor import BASProcessor
from mirrowrs.rivergeomproduct import RiverGeomProduct

# Input file
ex_dir = os.path.dirname(__file__)
watermask_tif = os.path.join(ex_dir, "example_watermask.tif")
ref_watermask_tif = os.path.join(ex_dir, "example_ref_waterbodies.shp")

# Simple example : sections are ready to use
shp_reaches_smpl = os.path.join(ex_dir, "example_reaches_simple.shp")
shp_sections_smpl = os.path.join(ex_dir, "example_sections_simple.shp")

# Complex example : sections have to be derived
shp_reaches_cplx = os.path.join(ex_dir, "example_reaches_cplx.shp")
shp_nodes_cplx = os.path.join(ex_dir, "example_nodes_cplx_up.shp")

# Load reference waterbodies - cfg 4-5
gdf_waterbodies = gpd.read_file(ref_watermask_tif)

# Load sections and reaches - cfg 1-4
gdf_reaches = gpd.read_file(shp_reaches_smpl)
gdf_sections = gpd.read_file(shp_sections_smpl)
gdf_sections.rename(mapper={"segment": "id"}, inplace=True, axis=1)


def example_1():
    """Example_1 :
    No watermask cleaning and no watermask labelling
    Sections available
    Width only
    """

    print("===== BASProcessing Example #1 = BEGIN =====")
    print("")

    # Set config #1
    dct_cfg_v1 = {
        "clean": {
            "bool_clean": False,
            "type_clean": "base",
            "fpath_wrkdir": ".",
            "gdf_waterbodies": None,
        },
        "label": {"bool_label": False, "type_label": "base", "fpath_wrkdir": "."},
        "reduce": {"how": "simple", "attr_nb_chan_max": None},
        "widths": {"scenario": 0},
    }

    # Instanciate basprocessor
    processor = BASProcessor(
        str_watermask_tif=watermask_tif,
        gdf_sections=gdf_sections,
        gdf_reaches=gdf_reaches,
        attr_reachid="id",
        str_proj="proj",
        str_provider="EO",
    )
    processor.preprocessing()
    processor.processing(dct_cfg_v1)
    gdf_widths, _ = processor.postprocessing(dct_cfg_v1)
    gdf_widths.to_file("widths_example1.shp")

    print("")
    print("===== BASProcessing Example #1 = END =====")


def example_2():
    """Example_2 :
    Basic watermask cleaning without reference waterbodies and no watermask labelling
    Sections available
    Width + estimation of intersection width other sections
    """

    print("===== BASProcessing Example #2 = BEGIN =====")
    print("")

    # Set config #2
    dct_cfg_v2 = {
        "clean": {
            "bool_clean": True,
            "type_clean": "base",
            "fpath_wrkdir": ".",
            "gdf_waterbodies": None,
        },
        "label": {"bool_label": False, "type_label": "base", "fpath_wrkdir": "."},
        "widths": {"scenario": 11},
    }

    # Instanciate basprocessor
    processor = BASProcessor(
        str_watermask_tif=watermask_tif,
        gdf_sections=gdf_sections,
        gdf_reaches=gdf_reaches,
        attr_reachid="id",
        str_proj="proj",
        str_provider="EO",
    )
    processor.preprocessing()
    processor.processing(dct_cfg_v2)
    gdf_widths, _ = processor.postprocessing(dct_cfg_v2)
    gdf_widths.to_file("widths_example2.shp")

    print("")
    print("===== BASProcessing Example #2 = END =====")


def example_3():
    """Example_3 :
    Watermask cleaning with reference waterbodies and no watermask labelling
    Sections available
    Width only
    """

    print("===== BASProcessing Example #3 = BEGIN =====")
    print("")

    # Set config #3
    dct_cfg_v3 = {
        "clean": {
            "bool_clean": True,
            "type_clean": "waterbodies",
            "fpath_wrkdir": ".",
            "gdf_waterbodies": gdf_waterbodies,
        },
        "label": {"bool_label": False, "type_label": "base", "fpath_wrkdir": "."},
        "widths": {"scenario": 0},
    }

    # Instanciate basprocessor
    processor = BASProcessor(
        str_watermask_tif=watermask_tif,
        gdf_sections=gdf_sections,
        gdf_reaches=gdf_reaches,
        attr_reachid="id",
        str_proj="proj",
        str_provider="EO",
    )
    processor.preprocessing()
    processor.processing(dct_cfg_v3)
    gdf_widths, _ = processor.postprocessing(dct_cfg_v3)
    gdf_widths.to_file("widths_example3.shp")

    print("")
    print("===== BASProcessing Example #3 = END =====")


def example_4():
    """Example_4 :
    Watermask cleaning with reference waterbodies + watermask labelling
    Sections available
    Width only
    """

    print("===== BASProcessing Example #4 = BEGIN =====")
    print("")

    # Set config #4
    dct_cfg_v4 = {
        "clean": {
            "bool_clean": True,
            "type_clean": "waterbodies",
            "fpath_wrkdir": ".",
            "gdf_waterbodies": gdf_waterbodies,
        },
        "label": {"bool_label": True, "type_label": "base", "fpath_wrkdir": "."},
        "widths": {"scenario": 0},
    }

    # Instanciate basprocessor
    processor = BASProcessor(
        str_watermask_tif=watermask_tif,
        gdf_sections=gdf_sections,
        gdf_reaches=gdf_reaches,
        attr_reachid="id",
        str_proj="proj",
        str_provider="EO",
    )
    processor.preprocessing()
    processor.processing(dct_cfg_v4)
    gdf_widths, _ = processor.postprocessing(dct_cfg_v4)
    gdf_widths.to_file("widths_example4.shp")

    print("")
    print("===== BASProcessing Example #4 = END =====")


def example_5():
    """Example_5 :
    Initialize reaches/nodes from shp and set projection from watermask
    """

    print("===== BASProcessing Example #5 = BEGIN =====")
    print("")

    dct_geom_attr = {
        "reaches": {"reaches_id": "reach_id"},
        "nodes": {
            "reaches_id": "reach_id",
            "nodes_id": "node_id",
            "pwidth": "p_width",
            "pwse": "p_wse",
        },
    }

    # Get watermask crs
    with rio.open(watermask_tif) as src:
        crs_wm_in = src.crs

    # Compute sections
    obj_rivergeom = RiverGeomProduct.from_shp(
        reaches_shp=shp_reaches_cplx,
        nodes_shp=shp_nodes_cplx,
        bool_edge=False,
        crs_in=crs_wm_in,
        dct_attr=dct_geom_attr,
    )
    obj_rivergeom.draw_allreaches_centerline()
    gdf_sections_ortho = obj_rivergeom.draw_allreaches_sections(
        type="ortho", flt_factor_width=15.0
    )

    # Compute sections
    obj_rivergeom = RiverGeomProduct.from_shp(
        reaches_shp=shp_reaches_cplx,
        nodes_shp=shp_nodes_cplx,
        bool_edge=False,
        dct_attr=dct_geom_attr,
    )
    obj_rivergeom.draw_allreaches_centerline()
    gdf_sections_ortho = obj_rivergeom.draw_allreaches_sections(
        type="ortho", flt_factor_width=15.0
    )

    print("")
    print("===== BASProcessing Example #5 = END =====")


def example_6():
    """Example_6 :
    Initialize reaches/nodes from shp
    Watermask cleaning with reference waterbodies + watermask labelling
    Sections NOT available
    Reduce section - "hydrogeom" + providing tolerance values as constant
    """

    print("===== BASProcessing Example #6 = BEGIN =====")
    print("")

    # Load reaches
    gdf_reaches_cplx = gpd.read_file(shp_reaches_cplx)
    gdf_nodes_cplx = gpd.read_file(shp_nodes_cplx)

    # Compute sections
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
        reaches_shp=shp_reaches_cplx,
        nodes_shp=shp_nodes_cplx,
        bool_edge=False,
        dct_attr=dct_geom_attr,
    )
    obj_rivergeom.draw_allreaches_centerline()
    gdf_sections_ortho = obj_rivergeom.draw_allreaches_sections(
        type="ortho", flt_factor_width=15.0
    )

    # Set configs #6
    dct_cfg_v6 = {
        "clean": {
            "bool_clean": True,
            "type_clean": "waterbodies",
            "fpath_wrkdir": "/home/cemery/Work/git/BAS/examples",
            "gdf_waterbodies": gdf_waterbodies,
        },
        "label": {
            "bool_label": True,
            "type_label": "base",
            "fpath_wrkdir": "/home/cemery/Work/git/BAS/examples",
        },
        "reduce": {
            "how": "hydrogeom",
            "attr_nb_chan_max": "n_chan_max",
            "attr_locxs": "loc_xs",
            "attr_nodepx": "x_proj",
            "attr_nodepy": "y_proj",
            "flt_tol_len": 0.05,
            "flt_tol_dist": 1000.0,
        },
        "widths": {"scenario": 11},
    }

    gdf_sections_ortho.insert(
        loc=2, column=dct_cfg_v6["reduce"]["attr_nb_chan_max"], value=0
    )
    gdf_sections_ortho[dct_cfg_v6["reduce"]["attr_nb_chan_max"]] = gdf_nodes_cplx.loc[
        gdf_sections_ortho.index, dct_cfg_v6["reduce"]["attr_nb_chan_max"]
    ]

    # Instanciate basprocessor(s)
    processor_a = BASProcessor(
        str_watermask_tif=watermask_tif,
        gdf_sections=gdf_sections_ortho,
        gdf_reaches=gdf_reaches_cplx,
        attr_reachid="reach_id",
        str_proj="proj",
        str_provider="EO",
    )
    processor_a.preprocessing()

    gser_proj_nodes = gdf_nodes_cplx["geometry"].to_crs(processor_a.watermask.crs)

    processor_a.gdf_sections.insert(
        loc=3, column=dct_cfg_v6["reduce"]["attr_nodepx"], value=0.0
    )
    processor_a.gdf_sections[dct_cfg_v6["reduce"]["attr_nodepx"]] = gser_proj_nodes.loc[
        processor_a.gdf_sections.index
    ].x

    processor_a.gdf_sections.insert(
        loc=4, column=dct_cfg_v6["reduce"]["attr_nodepy"], value=0.0
    )
    processor_a.gdf_sections[dct_cfg_v6["reduce"]["attr_nodepy"]] = gser_proj_nodes.loc[
        processor_a.gdf_sections.index
    ].y

    processor_a.processing(dct_cfg_v6)

    gdf_widths_a, _ = processor_a.postprocessing(dct_cfg_v6)

    gdf_widths_a["reach_id"] = gdf_widths_a["reach_id"].astype(str)
    gdf_widths_a["node_id"] = gdf_widths_a["node_id"].astype(int).astype(str)
    gdf_widths_a.to_file("widths_example6.shp")

    print("")
    print("===== BASProcessing Example #6 = END =====")


def example_7():
    """Example_7 :
    Initialize reaches/sections from gdf
    Watermask cleaning with reference waterbodies + watermask labelling
    Sections NOT available
    Reduce section - "hydrogeom" + providing tolerance values as constant
    """

    print("===== BASProcessing Example #7 = BEGIN =====")
    print("")

    # Load reaches
    gdf_reaches_cplx = gpd.read_file(shp_reaches_cplx)
    gdf_nodes_cplx = gpd.read_file(shp_nodes_cplx)

    # Compute sections
    dct_geom_attr = {
        "reaches": {"reaches_id": "reach_id"},
        "nodes": {
            "reaches_id": "reach_id",
            "nodes_id": "node_id",
            "pwidth": "p_width",
            "pwse": "p_wse",
        },
    }
    obj_rivergeom = RiverGeomProduct.from_gdf(
        gdf_reaches=gdf_reaches_cplx,
        gdf_nodes=gdf_nodes_cplx,
        bool_edge=False,
        dct_attr=dct_geom_attr,
    )
    obj_rivergeom.draw_allreaches_centerline()
    gdf_sections_ortho = obj_rivergeom.draw_allreaches_sections(
        type="ortho", flt_factor_width=15.0
    )

    # Set configs #7
    dct_cfg_v7 = {
        "clean": {
            "bool_clean": True,
            "type_clean": "waterbodies",
            "fpath_wrkdir": "/home/cemery/Work/git/BAS/examples",
            "gdf_waterbodies": gdf_waterbodies,
        },
        "label": {
            "bool_label": True,
            "type_label": "base",
            "fpath_wrkdir": "/home/cemery/Work/git/BAS/examples",
        },
        "reduce": {
            "how": "hydrogeom",
            "attr_nb_chan_max": "n_chan_max",
            "attr_locxs": "loc_xs",
            "attr_nodepx": "x_proj",
            "attr_nodepy": "y_proj",
            "flt_tol_len": 0.05,
            "flt_tol_dist": 1000.0,
        },
        "widths": {"scenario": 11},
    }

    gdf_sections_ortho.insert(
        loc=2, column=dct_cfg_v7["reduce"]["attr_nb_chan_max"], value=0
    )
    gdf_sections_ortho[dct_cfg_v7["reduce"]["attr_nb_chan_max"]] = gdf_nodes_cplx.loc[
        gdf_sections_ortho.index, dct_cfg_v7["reduce"]["attr_nb_chan_max"]
    ]

    # Instanciate basprocessor(s)
    processor_a = BASProcessor(
        str_watermask_tif=watermask_tif,
        gdf_sections=gdf_sections_ortho,
        gdf_reaches=gdf_reaches_cplx,
        attr_reachid="reach_id",
        str_proj="proj",
        str_provider="EO",
    )
    processor_a.preprocessing()

    gser_proj_nodes = gdf_nodes_cplx["geometry"].to_crs(processor_a.watermask.crs)

    processor_a.gdf_sections.insert(
        loc=3, column=dct_cfg_v7["reduce"]["attr_nodepx"], value=0.0
    )
    processor_a.gdf_sections[dct_cfg_v7["reduce"]["attr_nodepx"]] = gser_proj_nodes.loc[
        processor_a.gdf_sections.index
    ].x

    processor_a.gdf_sections.insert(
        loc=4, column=dct_cfg_v7["reduce"]["attr_nodepy"], value=0.0
    )
    processor_a.gdf_sections[dct_cfg_v7["reduce"]["attr_nodepy"]] = gser_proj_nodes.loc[
        processor_a.gdf_sections.index
    ].y

    processor_a.processing(dct_cfg_v7)

    gdf_widths_a, _ = processor_a.postprocessing(dct_cfg_v7)

    gdf_widths_a["reach_id"] = gdf_widths_a["reach_id"].astype(str)
    gdf_widths_a["node_id"] = gdf_widths_a["node_id"].astype(int).astype(str)
    gdf_widths_a.to_file("widths_example7.shp")

    print("")
    print("===== BASProcessing Example #7 = END =====")


def example_8():
    """Example_8 :
    Watermask cleaning with reference waterbodies + watermask labelling
    Sections NOT available
    Reduce section - "hydrogeom" + WITHOUT providing tolerance values
    """

    print("===== BASProcessing Example #8 = BEGIN =====")
    print("")

    # Load reaches
    gdf_reaches_cplx = gpd.read_file(shp_reaches_cplx)
    gdf_nodes_cplx = gpd.read_file(shp_nodes_cplx)

    # Compute sections
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
        reaches_shp=shp_reaches_cplx,
        nodes_shp=shp_nodes_cplx,
        bool_edge=False,
        dct_attr=dct_geom_attr,
    )
    obj_rivergeom.draw_allreaches_centerline()
    gdf_sections_ortho = obj_rivergeom.draw_allreaches_sections(
        type="ortho", flt_factor_width=15.0
    )

    # Set configs #8
    dct_cfg_v8 = {
        "clean": {
            "bool_clean": True,
            "type_clean": "waterbodies",
            "fpath_wrkdir": "/home/cemery/Work/git/BAS/examples",
            "gdf_waterbodies": gdf_waterbodies,
        },
        "label": {
            "bool_label": True,
            "type_label": "base",
            "fpath_wrkdir": "/home/cemery/Work/git/BAS/examples",
        },
        "reduce": {
            "how": "hydrogeom",
            "attr_nb_chan_max": "n_chan_max",
            "attr_locxs": "loc_xs",
            "attr_nodepx": "x_proj",
            "attr_nodepy": "y_proj",
        },
        "widths": {"scenario": 11},
    }

    gdf_sections_ortho.insert(
        loc=2, column=dct_cfg_v8["reduce"]["attr_nb_chan_max"], value=0
    )
    gdf_sections_ortho[dct_cfg_v8["reduce"]["attr_nb_chan_max"]] = gdf_nodes_cplx.loc[
        gdf_sections_ortho.index, dct_cfg_v8["reduce"]["attr_nb_chan_max"]
    ]

    # Instanciate basprocessor(s)
    processor_a = BASProcessor(
        str_watermask_tif=watermask_tif,
        gdf_sections=gdf_sections_ortho,
        gdf_reaches=gdf_reaches_cplx,
        attr_reachid="reach_id",
        str_proj="proj",
        str_provider="EO",
    )
    processor_a.preprocessing()

    gser_proj_nodes = gdf_nodes_cplx["geometry"].to_crs(processor_a.watermask.crs)

    processor_a.gdf_sections.insert(
        loc=3, column=dct_cfg_v8["reduce"]["attr_nodepx"], value=0.0
    )
    processor_a.gdf_sections[dct_cfg_v8["reduce"]["attr_nodepx"]] = gser_proj_nodes.loc[
        processor_a.gdf_sections.index
    ].x

    processor_a.gdf_sections.insert(
        loc=4, column=dct_cfg_v8["reduce"]["attr_nodepy"], value=0.0
    )
    processor_a.gdf_sections[dct_cfg_v8["reduce"]["attr_nodepy"]] = gser_proj_nodes.loc[
        processor_a.gdf_sections.index
    ].y

    processor_a.processing(dct_cfg_v8)

    gdf_widths_a, _ = processor_a.postprocessing(dct_cfg_v8)

    gdf_widths_a["reach_id"] = gdf_widths_a["reach_id"].astype(str)
    gdf_widths_a["node_id"] = gdf_widths_a["node_id"].astype(int).astype(str)
    gdf_widths_a.to_file("widths_example8.shp")

    print("")
    print("===== BASProcessing Example #8 = END =====")


def example_9():
    """Example_9 :
    Watermask cleaning with reference waterbodies + watermask labelling
    Sections NOT available
    Reduce : base
    2 width products over the same mask
    """

    print("===== BASProcessing Example #9 = BEGIN =====")
    print("")

    # Load reaches
    gdf_reaches_cplx = gpd.read_file(shp_reaches_cplx)

    # Compute sections
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
        reaches_shp=shp_reaches_cplx,
        nodes_shp=shp_nodes_cplx,
        bool_edge=False,
        dct_attr=dct_geom_attr,
    )
    obj_rivergeom.draw_allreaches_centerline()

    gdf_sections_ortho = obj_rivergeom.draw_allreaches_sections(
        type="ortho", flt_factor_width=15.0
    )
    gdf_sections_ortho.to_file("ex6_sections_ortho.shp")

    gdf_sections_chck = obj_rivergeom.draw_allreaches_sections(type="chck")
    gdf_sections_chck.to_file("ex6_sections_chck.shp")

    # Set configs #9
    dct_cfg_v9 = {
        "clean": {
            "bool_clean": True,
            "type_clean": "waterbodies",
            "fpath_wrkdir": "/home/cemery/Work/git/BAS/examples",
            "gdf_waterbodies": gdf_waterbodies,
        },
        "label": {
            "bool_label": True,
            "type_label": "base",
            "fpath_wrkdir": "/home/cemery/Work/git/BAS/examples",
        },
        "widths": {"scenario": 11},
    }

    dct_cfg_v9b = {
        "clean": {
            "bool_clean": False,
            "type_clean": "waterbodies",
            "fpath_wrkdir": "/home/cemery/Work/git/BAS/examples",
            "gdf_waterbodies": gdf_waterbodies,
        },
        "label": {
            "bool_label": False,
            "type_label": "base",
            "fpath_wrkdir": "/home/cemery/Work/git/BAS/examples",
        },
        "widths": {"scenario": 0},
    }

    # Instanciate basprocessor(s)
    processor_a = BASProcessor(
        str_watermask_tif=watermask_tif,
        gdf_sections=gdf_sections_ortho,
        gdf_reaches=gdf_reaches_cplx,
        attr_reachid="reach_id",
        str_proj="proj",
        str_provider="EO",
    )
    processor_a.preprocessing()

    processor_a.processing(dct_cfg_v9)

    gdf_widths_a, str_fpath_updated_wm_tif = processor_a.postprocessing(dct_cfg_v9)
    gdf_widths_a["reach_id"] = gdf_widths_a["reach_id"].astype(str)
    gdf_widths_a["node_id"] = gdf_widths_a["node_id"].astype(int).astype(str)
    gdf_widths_a.to_file("widths_a_example9.shp")

    processor_b = BASProcessor(
        str_watermask_tif=str_fpath_updated_wm_tif,
        gdf_sections=gdf_sections_chck,
        gdf_reaches=gdf_reaches_cplx,
        attr_reachid="reach_id",
        str_proj="proj",
        str_provider="EO",
    )

    processor_b.preprocessing()

    processor_b.processing(dct_cfg_v9b)

    dct_cfg_v9b["clean"]["bool_clean"] = True
    dct_cfg_v9b["label"]["bool_label"] = True
    gdf_widths_b, _ = processor_b.postprocessing(dct_cfg_v9b)

    gdf_widths_b["reach_id"] = gdf_widths_b["reach_id"].astype(str)
    gdf_widths_b["node_id"] = gdf_widths_b["node_id"].astype(int).astype(str)
    gdf_widths_b.to_file("widths_b_example9.shp")

    print("")
    print("===== BASProcessing Example #9 = END =====")


def example_10():
    """Example_10 :
    Watermask cleaning base + watermask labelling
    Sections NOT available
    Reduce section - "hydrogeom" + providing tolerance values as series
    """

    print("===== BASProcessing Example #10 = BEGIN =====")
    print("")

    # Load reaches
    gdf_reaches_cplx = gpd.read_file(shp_reaches_cplx)
    gdf_nodes_cplx = gpd.read_file(shp_nodes_cplx)

    # Compute sections
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
        reaches_shp=shp_reaches_cplx,
        nodes_shp=shp_nodes_cplx,
        bool_edge=False,
        dct_attr=dct_geom_attr,
    )
    obj_rivergeom.draw_allreaches_centerline()
    gdf_sections_ortho = obj_rivergeom.draw_allreaches_sections(
        type="ortho", flt_factor_width=15.0
    )

    # Set configs #10
    dct_cfg_v10 = {
        "clean": {
            "bool_clean": True,
            "type_clean": "waterbodies",
            "fpath_wrkdir": ".",
            "gdf_waterbodies": gdf_waterbodies,
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

    # Add specific attributes
    attr_nb_chan_max = dct_cfg_v10["reduce"]["attr_nb_chan_max"]
    gdf_sections_ortho.insert(loc=2, column=attr_nb_chan_max, value=0)
    gdf_sections_ortho[attr_nb_chan_max] = gdf_nodes_cplx.loc[
        gdf_sections_ortho.index, attr_nb_chan_max
    ]

    attr_meander_length = dct_cfg_v10["reduce"]["attr_meander_length"]
    attr_sinuosity = dct_cfg_v10["reduce"]["attr_sinuosity"]
    attr_tol_dist = dct_cfg_v10["reduce"]["flt_tol_dist"]
    gdf_sections_ortho.insert(loc=3, column=attr_tol_dist, value=0.0)
    gdf_sections_ortho[attr_tol_dist] = (
        0.5
        * gdf_nodes_cplx.loc[gdf_sections_ortho.index, attr_meander_length]
        / gdf_nodes_cplx.loc[gdf_sections_ortho.index, attr_sinuosity]
    )

    # Instanciate basprocessor(s)
    processor_a = BASProcessor(
        str_watermask_tif=watermask_tif,
        gdf_sections=gdf_sections_ortho,
        gdf_reaches=gdf_reaches_cplx,
        attr_reachid="reach_id",
        str_proj="proj",
        str_provider="EO",
    )
    processor_a.preprocessing()

    gser_proj_nodes = gdf_nodes_cplx["geometry"].to_crs(processor_a.watermask.crs)

    processor_a.gdf_sections.insert(
        loc=3, column=dct_cfg_v10["reduce"]["attr_nodepx"], value=0.0
    )
    processor_a.gdf_sections[dct_cfg_v10["reduce"]["attr_nodepx"]] = gser_proj_nodes.loc[
        processor_a.gdf_sections.index
    ].x

    processor_a.gdf_sections.insert(
        loc=4, column=dct_cfg_v10["reduce"]["attr_nodepy"], value=0.0
    )
    processor_a.gdf_sections[dct_cfg_v10["reduce"]["attr_nodepy"]] = gser_proj_nodes.loc[
        processor_a.gdf_sections.index
    ].y

    processor_a.processing(dct_cfg_v10)

    gdf_widths_a, _ = processor_a.postprocessing(dct_cfg_v10)

    gdf_widths_a["reach_id"] = gdf_widths_a["reach_id"].astype(str)
    gdf_widths_a["node_id"] = gdf_widths_a["node_id"].astype(int).astype(str)
    gdf_widths_a.to_file("widths_example10.shp")

    print("")
    print("===== BASProcessing Example #10 = END =====")


def main():
    """Main run
    """

    # Run example 1
    try:
        example_1()
    except:
        print("Fail example 1")

    # Run example 2
    try:
        example_2()
    except:
        print("Fail example 2")

    # Run example 3
    try:
        example_3()
    except:
        print("Fail example 3")

    # Run example 4
    try:
        example_4()
    except:
        print("Fail example 4")

    # Run example 5
    try:
        example_5()
    except:
        print("Fail example 5")

    # Run example 6
    try:
        example_6()
    except:
        print("Fail example 6")

    # Run example 7
    try:
        example_7()
    except:
        print("Fail example 7")

    # Run example 8
    try:
        example_8()
    except:
        print("Fail example 8")

    # Run example 9
    try:
        example_9()
    except:
        print("Fail example 9")

    # Run example 10
    try:
        example_10()
    except:
        print("Fail example 10")


if __name__ == "__main__":
    main()
