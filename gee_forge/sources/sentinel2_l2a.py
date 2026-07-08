from __future__ import annotations

import ee

from gee_forge.sources.base import ImageSource

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

def mask_s2_clouds(image: ee.Image):
    """Mask invalid and cloudy Sentinel-2 pixels using the SCL band.

    Args:
        image: Sentinel-2 L2A image containing the ``SCL`` scene
            classification band.

    Returns:
        Input image with cloud, shadow, snow, and invalid SCL classes masked.
    """
    cloud_mask = image.select('SCL').eq(0).Or(image.select('SCL').eq(1)).Or(image.select('SCL').eq(2)).Or(image.select('SCL').eq(3)).Or(image.select('SCL').eq(8)).Or(image.select('SCL').eq(9)).Or(image.select('SCL').eq(10))
    return image.updateMask(cloud_mask.Not())


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
        image = (
            collection.map(ee.Image.resample)
            .map(mask_s2_clouds)
            .select(S2L2A_BANDS)
            .median()
            .reproject(crs=crs, crsTransform=crs_transform)
        )

        if window_name is None:
            return image
        return image.rename([f"{band}_{window_name}" for band in S2L2A_BANDS])
