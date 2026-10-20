from __future__ import annotations

from typing import Any

import ee

from gee_forge.sources.base import ImageSource, count_nodata

S2L2A_BANDS = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
        ]

# SCL classes treated as invalid: no data, saturated/defective, dark area,
# cloud shadow, cloud medium probability, cloud high probability, cirrus.
SCL_INVALID_CLASSES = [0, 1, 2, 3, 8, 9, 10]

def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Mask invalid Sentinel-2 pixels using the SCL band.

    SCL is categorical, so it must keep nearest-neighbour resampling. Call this
    before setting bilinear resampling on the reflectance bands.
    """
    scl = image.select("SCL")
    invalid = scl.remap(
        SCL_INVALID_CLASSES,
        [1] * len(SCL_INVALID_CLASSES),
        defaultValue=0,
    )
    return image.updateMask(invalid.Not())

def prepare_s2(image: ee.Image) -> ee.Image:
    """Mask with SCL, then interpolate only the reflectance bands."""
    return mask_s2_clouds(image).select(S2L2A_BANDS).resample("bilinear")


def blank_s2(collection_id: str) -> ee.Image:
    """Build a fully masked scene that keeps empty windows well formed.

    A window can end up with no scenes at all, and the median of an empty
    collection is an image with no bands: renaming it fails, and so does
    exporting it next to the windows that do have data. Merging this image
    into every window keeps the band names and the band types of a real
    scene. Every pixel of it is masked, so it never reaches the median of a
    window that has scenes of its own, and a window that has none composites
    to a chip that is masked everywhere.

    Args:
        collection_id: Collection the window scenes come from, used so the
            blank scene carries exactly their band types.

    Returns:
        Prepared scene with every pixel masked.
    """
    scene = ee.Image(ee.ImageCollection(collection_id).first())
    return prepare_s2(scene).updateMask(ee.Image.constant(0))


class S2L2ASource(ImageSource):
    """Sentinel-2 L2A source using the original median-composite pipeline."""

    def __init__(
        self,
        cloud_limit: int = 20,
        snowice_limit: int = 20,
        global_filter: ee.Filter | list[ee.Filter] | None = None,
        initialize_ee: bool = True,
    ):
        filters = [
            ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_limit),
            ee.Filter.lt("SNOW_ICE_PERCENTAGE", snowice_limit),
        ]
        if global_filter is not None:
            filters.extend(global_filter if isinstance(global_filter, list) else [global_filter])

        super().__init__(
            ee_collection="COPERNICUS/S2_SR_HARMONIZED",
            name="S2L2A",
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
        prepared = collection.map(prepare_s2).merge(
            ee.ImageCollection([blank_s2(self.ee_collection_id)])
        )
        image = prepared.median().reproject(crs=crs, crsTransform=crs_transform)

        if window_name is None:
            return image
        return image.rename([f"{band}_{window_name}" for band in S2L2A_BANDS])

    def chip_metadata(
        self,
        image: ee.Image,
        collection: ee.ImageCollection,
        window_name: str | None,
        time_bin: str,
        region: ee.Geometry,
        crs: str,
        crs_transform: list[float],
        width: int,
        height: int,
    ) -> dict[str, Any]:
        """Record how much of the window composite holds no valid data.

        Every reflectance band carries the one SCL-derived mask, so counting a
        single band describes the whole chip. A window with no scenes
        composites to a masked chip, and is therefore reported as missing in
        all of its pixels without needing a case of its own.
        """
        counts = count_nodata(
            image=image,
            region=region,
            crs=crs,
            crs_transform=crs_transform,
        )
        nodata = ee.Number(counts.get("nodata")).round().toInt()
        counted = ee.Number(counts.get("counted")).round().toInt()

        window_key = window_name if window_name is not None else self.name.lower()
        return {
            window_key: ee.Dictionary(
                {
                    "nodata_pixels": nodata,
                    "nodata_fraction": nodata.divide(width * height),
                    # Equal to the chip's pixel count unless Earth Engine
                    # reduced on a grid other than the chip's own.
                    "counted_pixels": counted,
                    "scene_count": collection.size(),
                }
            )
        }
