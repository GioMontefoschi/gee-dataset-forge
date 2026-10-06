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
    """USDA CDL source producing the original binary label output.

    CDL is an annual product, so this source collapses the configured time
    windows into the single calendar year they cover instead of producing one
    output per window.
    """

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

    def resolve_year(self, time_windows: dict[str, tuple[str, str]]) -> int:
        """Return the CDL calendar year covered by the configured windows.

        CDL publishes one image per year, stamped ``{year}-01-01``. Earth
        Engine date filters match on ``system:time_start``, so a window that
        does not contain a January 1st matches no image at all. The year is
        therefore read from the window dates and expanded to a full calendar
        year, rather than used as a filter range directly.

        The year is taken from the earliest window start. Child classes can
        override this when windows span two years and the later year is wanted.

        Args:
            time_windows: Mapping of ``window_name`` to
                ``(start_date, end_date)``. Never empty.

        Returns:
            Calendar year selected from the configured windows.
        """
        earliest_start = min(start_date for start_date, _ in time_windows.values())
        return int(earliest_start[:4])

    def filter_time_windows(
        self,
        collection: ee.ImageCollection,
        time_windows: dict[str, tuple[str, str]],
        region: ee.Geometry,
    ):
        """Yield one annual collection instead of one output per window."""
        if not time_windows:
            yield None, collection
            return

        year = self.resolve_year(time_windows)
        yield None, collection.filterDate(f"{year}-01-01", f"{year + 1}-01-01")

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
