# gee-dataset-forge

`gee-dataset-forge` is a Python package for building remote-sensing chip
datasets from Google Earth Engine. It focuses on the acquisition layer:

- generate spatially distributed sample points from a binary label source;
- download spatially aligned GeoTIFF chips for those points;
- combine multiple Earth Engine sources such as Sentinel-2, Sentinel-1, DEM,
  satellite embeddings, and USDA CDL labels.

The package is intentionally limited to Earth Engine sampling and export. Model
training, TerraTorch integration, and downstream dataset preparation should live
outside this repository.

## Status

This is an early research codebase. The public import package is `gee_forge`.
The repository currently provides a Python API; it does not provide a stable
command-line interface.

## Installation

Use Python 3.12 or newer.

```bash
python -m pip install .
```

## Earth Engine Setup

Authenticate the Earth Engine CLI before running live examples:

```bash
earthengine authenticate
```

Most workflows should initialize Earth Engine with an explicit Google Cloud
project. The package exposes the high-volume Earth Engine endpoint used by the
downloader:

```python
import ee

from gee_forge.downloader import EARTH_ENGINE_HIGH_VOLUME_URL

ee.Initialize(
    project="your-google-cloud-project",
    opt_url=EARTH_ENGINE_HIGH_VOLUME_URL,
)
```

The downloader also passes the project and endpoint into worker processes when
`ImageSourceDownloader` is configured with `ee_project` and `ee_opt_url`.

## Core Concepts

### Label Sources

Sampling is driven by a `LabelSource`. A label source owns the Earth Engine
label image and the metadata needed by the sampler:

- sampling band name;
- positive class value;
- other/background class value;
- sampling scale;
- optional projection;
- optional remapping logic.

`CDLBinaryLabelSource` is the built-in USDA CDL preset. It remaps the CDL
`cropland` band to a binary `label` band where:

- `2` means cultivated;
- `1` means non-cultivated.

### Sampling

`MontefSampling` tiles an ROI into projected grid cells, draws one candidate
point per valid cell from the positive class, and keeps only points whose chip
footprint contains enough positive pixels.

The sampler returns an `ee.FeatureCollection`.

### Image Sources

An `ImageSource` describes how to turn one Earth Engine collection into one
chip image for a point and time bin. Built-in sources are:

- `S2L2ASource`: Sentinel-2 L2A median composites from
  `COPERNICUS/S2_SR_HARMONIZED`;
- `S1GRDSource`: Sentinel-1 GRD VV/VH median composites from
  `COPERNICUS/S1_GRD`;
- `DEMSource`: Copernicus GLO-30 DEM mosaic from `COPERNICUS/DEM/GLO30`;
- `AEFSource`: annual satellite embeddings from
  `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`;
- `CDLSource`: binary USDA CDL label chips from `USDA/NASS/CDL`.

### Downloader

`ImageSourceDownloader` expands sample points, image sources, time bins, and
time windows into download requests. It writes GeoTIFF chips to:

```text
{out_dir}/{time_bin}/{source_name}/{point_id}_{time_bin}_{source_name}.tif
```

For example:

```text
dataset_chips/2022/s2l2a/000_2022_s2l2a.tif
```

## Example: Sample Points From CDL

```python
import ee

from gee_forge.downloader import EARTH_ENGINE_HIGH_VOLUME_URL
from gee_forge.sampling import CDLBinaryLabelSource, MontefSampling

EE_PROJECT = "your-google-cloud-project"

ee.Initialize(project=EE_PROJECT, opt_url=EARTH_ENGINE_HIGH_VOLUME_URL)

roi = ee.Geometry.Rectangle(
    [-94.10, 41.90, -93.40, 42.40],
    proj="EPSG:4326",
    geodesic=False,
)

label_source = CDLBinaryLabelSource(year=2022)

sampler = MontefSampling(
    roi=roi,
    label_source=label_source,
    grid_size=8_000,
    buffer=2_240,
    min_sample_fraction=0.1,
    seed=42,
)

points = sampler.sample()
print(points.size().getInfo())
```

In `MontefSampling`, `buffer` is the desired chip side length in meters. The
acceptance check uses half of that distance around each candidate point.

## Example: Custom Binary Label Source

Use `LabelSource` directly when the label image is not one of the built-in
presets.

```python
import ee

from gee_forge.sampling import LabelSource, MontefSampling

roi = ee.Geometry.Rectangle(
    [-94.10, 41.90, -93.40, 42.40],
    proj="EPSG:4326",
    geodesic=False,
)

label_image = ee.Image("users/example/binary_label").rename("label")

label_source = LabelSource(
    image=label_image,
    band_name="label",
    sample_value=1,
    other_value=0,
    sample_scale=30,
)

points = MontefSampling(
    roi=roi,
    label_source=label_source,
    grid_size=5_000,
    buffer=1_000,
    min_sample_fraction=0.25,
).sample()
```

`LabelSource.image` may also be a callable that receives the ROI and returns an
`ee.Image`. That is useful when a label collection needs to be filtered by the
current ROI before selecting an image.

## Example: Download Image Chips

```python
from pathlib import Path

import ee

from gee_forge.downloader import EARTH_ENGINE_HIGH_VOLUME_URL, ImageSourceDownloader
from gee_forge.sources import AEFSource, DEMSource, S1GRDSource, S2L2ASource

EE_PROJECT = "your-google-cloud-project"
OUT_DIR = Path("dataset_chips")

ee.Initialize(project=EE_PROJECT, opt_url=EARTH_ENGINE_HIGH_VOLUME_URL)

point = ee.Geometry.Point([-93.75, 42.15])
points = [
    (
        "000",
        {
            "type": "Point",
            "coordinates": [-93.75, 42.15],
        },
    )
]

projection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(point)
    .filterDate("2022-01-01", "2023-01-01")
    .first()
    .select("B2")
    .projection()
    .getInfo()
)

sources = [
    S2L2ASource(cloud_limit=20, snowice_limit=20),
    S1GRDSource(),
    DEMSource(),
    AEFSource(),
]

downloader = ImageSourceDownloader(
    points=points,
    image_sources=sources,
    ee_project=EE_PROJECT,
    ee_opt_url=EARTH_ENGINE_HIGH_VOLUME_URL,
)

downloader.bulk_download(
    out_dir=OUT_DIR,
    crs=projection["crs"],
    crs_transform=projection["transform"],
    dimensions="224x224",
    time_bins=["2022"],
    time_windows={
        "spring": ("2022-04-01", "2022-06-01"),
        "summer": ("2022-06-01", "2022-09-01"),
        "fall": ("2022-09-01", "2022-11-01"),
    },
    num_workers=4,
    show_progress=True,
)
```

### Chip Geometry

Chip extent comes from `crs_transform` and `dimensions`: a `224x224` chip on a
`10` meter transform covers `2_240` meters. There is no separate buffer.

Each point gets its own `crsTransform`, centred on the point and snapped to the
lattice of `crs_transform`, and that same transform is used both to reproject the
image and to request the download. Chips therefore land on the reference grid
exactly, and co-register across points, dates, and sources. Snapping moves the
chip centre by up to half a pixel, which is the cost of keeping pixels aligned.

Do not pass `region` plus `dimensions` to `getDownloadURL` instead. Earth Engine
then derives its own transform from the region bounds, and because a lat/lon
bounding box is rotated relative to a projected grid, the derived pixel size
drifts off the nominal scale. Resampling onto that unaligned grid drops a row or
column every `scale / drift` pixels and leaves a visible blocky seam pattern in
every band.

`crs` is used for all points, so a point set spanning several UTM zones needs to
be split by zone and downloaded once per zone.

`filter_margin` is optional and defaults to `0`. It widens only the region used
to filter source collections, never the output grid.

`points` may be provided as:

- an `ee.FeatureCollection`;
- a raw GeoJSON-like point geometry;
- a list of GeoJSON-like point geometries;
- `(point_id, geometry)` tuples;
- dictionaries with `id` plus `geometry`, `point`, `.geo`, or `geo`.

## Time Bins and Time Windows

`time_bins` control the output directory and file-name grouping. `time_windows`
control temporal filtering inside each source.

With:

```python
time_bins = ["2022"]
time_windows = {
    "spring": ("2022-04-01", "2022-06-01"),
    "summer": ("2022-06-01", "2022-09-01"),
}
```

Sentinel-2 and Sentinel-1 create one composite per window and concatenate the
window bands into a single chip. Windowed band names are suffixed with the
window name, such as `B2_spring` or `VV_summer`.

Sources can override this behavior. `DEMSource`, for example, ignores time
windows because elevation is static.

Annual sources collapse the configured windows instead. `CDLSource` produces a
single `label` band per chip, selecting the calendar year that the windows fall
in. This is necessary because Earth Engine date filters match on
`system:time_start`, and every CDL image is stamped `{year}-01-01`: a window
such as `("2022-04-01", "2022-06-01")` contains no January 1st and so would
match no image at all. The year is taken from the earliest window start, which
for windows spanning two years means the earlier year is used. Override
`CDLSource.resolve_year()` to select the later year instead:

```python
class WinterCDLSource(CDLSource):
    def resolve_year(self, time_windows: dict[str, tuple[str, str]]) -> int:
        return super().resolve_year(time_windows) + 1
```

## Creating a New Image Source

Add a new source by subclassing `ImageSource`. Most sources only need to set the
Earth Engine collection ID, output name, optional global filters, and
`preprocess()`.

```python
from __future__ import annotations

import ee

from gee_forge.sources import ImageSource


class NDVISource(ImageSource):
    def __init__(
        self,
        global_filter: ee.Filter | list[ee.Filter] | None = None,
        initialize_ee: bool = True,
    ):
        filters = [ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)]
        if global_filter is not None:
            filters.extend(
                global_filter if isinstance(global_filter, list) else [global_filter]
            )

        super().__init__(
            ee_collection="COPERNICUS/S2_SR_HARMONIZED",
            name="NDVI",
            global_filter=filters,
            initialize_ee=initialize_ee,
        )

    def preprocess(
        self,
        collection: ee.ImageCollection,
        window_name: str | None,
        time_bin: str,
        region: ee.Geometry,
        crs: str,
        crs_transform: list[float],
    ) -> ee.Image:
        image = collection.median()
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndvi = ndvi.reproject(crs=crs, crsTransform=crs_transform)

        if window_name is None:
            return ndvi
        return ndvi.rename(f"NDVI_{window_name}")
```

Override `filter_time_window()` when a collection cannot use a plain
`filterDate(start_date, end_date)` call. Override `filter_time_windows()` only
when a source should not produce one output per configured window.

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for the
full license text.
