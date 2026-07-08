from __future__ import annotations

import ee

from gee_forge.sources.base import ImageSource


class DEMSource(ImageSource):
    """Copernicus DEM source using the original DEM mosaic pipeline."""

    def __init__(
        self,
        global_filter: ee.Filter | list[ee.Filter] | None = None,
        initialize_ee: bool = True,
    ):
        super().__init__(
            ee_collection="COPERNICUS/DEM/GLO30",
            name="DEM",
            global_filter=global_filter,
            initialize_ee=initialize_ee,
        )

    def filter_time_windows(
        self,
        collection: ee.ImageCollection,
        time_windows: dict[str, tuple[str, str]],
        region: ee.Geometry,
    ):
        yield None, collection

    def preprocess(
        self,
        collection: ee.ImageCollection,
        window_name: str | None,
        time_bin: str,
        region: ee.Geometry,
        crs: str,
        crs_transform: list[float],
    ) -> ee.Image:
        return (
            collection.map(ee.Image.resample)
            .select("DEM")
            .mosaic()
            .reproject(crs=crs, crsTransform=crs_transform)
        )
