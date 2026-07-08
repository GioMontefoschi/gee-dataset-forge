from __future__ import annotations

import ee

from gee_forge.sources.base import ImageSource


S1GRD_BANDS = [
    "VV",
    "VH"
    ]


class S1GRDSource(ImageSource):
    """Sentinel-1 GRD source using the original median-composite pipeline."""

    def __init__(
        self,
        global_filter: ee.Filter | list[ee.Filter] | None = None,
        initialize_ee: bool = True,
    ):
        filters = [
            ee.Filter.eq("instrumentMode", "IW"),
            ee.Filter.listContains("transmitterReceiverPolarisation", "VV"),
            ee.Filter.listContains("transmitterReceiverPolarisation", "VH"),
        ]
        if global_filter is not None:
            filters.extend(global_filter if isinstance(global_filter, list) else [global_filter])

        super().__init__(
            ee_collection="COPERNICUS/S1_GRD",
            name="S1GRD",
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
            .select(S1GRD_BANDS)
            .median()
            .reproject(crs=crs, crsTransform=crs_transform)
        )

        if window_name is None:
            return image
        return image.rename([f"{band}_{window_name}" for band in S1GRD_BANDS])
