from __future__ import annotations

import ee

from gee_forge.sources.base import ImageSource


class AEFSource(ImageSource):
    """Annual satellite embedding source using the original mosaic pipeline."""

    def __init__(
        self,
        global_filter: ee.Filter | list[ee.Filter] | None = None,
        initialize_ee: bool = True,
    ):
        super().__init__(
            ee_collection="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
            name="AEF",
            global_filter=global_filter,
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
        return collection.mosaic().reproject(crs=crs, crsTransform=crs_transform)
