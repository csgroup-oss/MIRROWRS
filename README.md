# MIRROWRS

<h4> Mapper to InfeR River Observations of Widths from Remote Sensing: compute river surface width from a watermask image </h4> 

## Overview

Pronounce it "mirrors" !

MIRROWRS is a simple Python-based toolbox designed to compute river widths from a space-borne watermask given a set of section (line geometries) along the river.


## Download

Main version available on github : https://github.com/csgroup-oss/MIRROWRS

## Installation

### General

Using a virtual environment is strongly recommended.

#### Locally

If you have python-3.12 and `sw1Dto2D` is already installed, then running this command will install BAS requirements

```bash
# if gdal is not yet installed 
pip install GDAL==`gdal-config --version`

pip install -e .

# then try to access the entry point
run_bas -h
```

This entrypoint is to compute BAS widths on Surfwater-like watermasks.

## Examples

To run examples, use the entrypoint

```bash
run_examples
```

## Usage

To use BAS on Surfwater-like watermasks, here is the general command

```bash
run_bas -w /path/to/water_mask.tif -dt YYYYmmddThhmmss -r /path/to/eu_sword_reaches_hb23_v16.shp -n /path/to/eu_sword_nodes_hb23_v16.shp -o /path/to/output/directory
```

This will only save a csv file with widths data. You can had `--more_outputs` to also save the cleaned watermask and the shapefile.  


## Features

Using the BAS toolbox, you can perform two tasks :

- Given a watermask (as a GeoTiff file), a set on centerline reaches (as a shapefile of LineString) and a set of sections (as a shapefile of LineString),
derive the river widths along the sections
- If the sections lines are not available, using the centerline reaches (as a shapefile of LineString) and a set of segmentation points (as a shapefile of Point),
you can draw the sections yourself


## License

BAS is licensed by [CS GROUP](https://www.c-s.fr/) under
the [Apache License, version 2.0](http://www.apache.org/licenses/LICENSE-2.0.html).
A copy of this license is provided in the [LICENSE](LICENSE) file.