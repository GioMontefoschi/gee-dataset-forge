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
            collection.map(prepare_s2)
            .median()
            .reproject(crs=crs, crsTransform=crs_transform)
        )

        if window_name is None:
            return image
        return image.rename([f"{band}_{window_name}" for band in S2L2A_BANDS])
