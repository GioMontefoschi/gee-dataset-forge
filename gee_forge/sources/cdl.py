from __future__ import annotations

import ee

from gee_forge.sources.base import ImageSource


def remap_crops(img: ee.Image):
    """Remap a USDA CDL image to binary cultivated/non-cultivated labels.

    Args:
        img: Image from the ``USDA/NASS/CDL`` collection with a ``cropland``
            band.

    Returns:
        Image with a single ``label`` band where ``1`` is non-cultivated and
        ``2`` is cultivated.
    """
    from_values = [0] + [i for i in range(1, 61)] + [i for i in range(61, 66)] + [i for i in range(66, 81)] + [i for i in range(81, 196)] + [i for i in range(196, 256)]
    to_values = [1] + [2 for i in range(1, 61)] + [1 for i in range(61, 66)] + [2 for i in range(66, 81)] + [1 for i in range(81, 196)] + [2 for i in range(196, 256)]
    return img.remap(from_values, to_values, defaultValue=0, bandName='cropland').select('remapped').rename('label')


class CDLSource(ImageSource):
    """USDA CDL source producing the original binary label output."""

    def __init__(
        self,
        global_filter: ee.Filter | list[ee.Filter] | None = None,
        initialize_ee: bool = True,
    ):
        super().__init__(
            ee_collection="USDA/NASS/CDL",
            name="LABEL",
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
        image = remap_crops(ee.Image(collection.first()))
        return image.reproject(crs=crs, crsTransform=crs_transform)
