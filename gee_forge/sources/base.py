from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable

import ee
import requests as http_requests
from retry import retry


class ImageSource:
    """Base class for Earth Engine image sources.

    The class owns the common request generation and download pipeline. Child
    classes can override ``filter_time_window`` and ``preprocess`` for
    collection-specific temporal logic and image preparation.
    """

    def __init__(
        self,
        ee_collection: str,
        name: str,
        global_filter: ee.Filter | list[ee.Filter] | None = None,
        initialize_ee: bool = True,
    ):
        """Initialize an image source.

        Args:
            ee_collection: Earth Engine image collection id.
            name: Source name used in request ids and output paths.
            global_filter: Earth Engine filter or filters applied to the full
                collection before spatial and temporal filtering.
            initialize_ee: If ``True``, worker calls initialize Earth Engine
                before requesting a download URL.
        """
        self.name = name
        self.ee_collection_id = ee_collection
        self.global_filter = self._normalize_global_filter(global_filter)
        self.initialize_ee = initialize_ee

    def get_requests(
        self,
        id_point_geometries: Iterable[Any],
        buffer: int | float,
        crs: str,
        crs_transform: list[float],
        dimensions: str,
        time_bins: str | Iterable[str],
        time_windows: dict[str, tuple[str, str]] | None,
        out_dir: str | Path,
    ) -> list[tuple]:
        """Generate download request tuples for this source.

        Args:
            id_point_geometries: Iterable containing point ids and GeoJSON-like
                point geometries. Supported items are ``(id, geometry)`` tuples,
                dicts with ``id`` plus ``geometry``/``point``/``.geo``, or raw
                point geometries, which are enumerated.
            buffer: Point buffer, in meters, used to build chip bounds.
            crs: Output coordinate reference system.
            crs_transform: Output affine transform passed to Earth Engine as
                ``crsTransform``.
            dimensions: Output chip dimensions as ``"{width}x{height}"``.
            time_bins: Output bin or bins used to organize downloaded files.
            time_windows: Mapping of ``window_name`` to ``(start_date, end_date)``.
            out_dir: Root directory where chips will be written.

        Returns:
            Request tuples that can be consumed by ``get_result``.
        """
        time_bins = [time_bins] if isinstance(time_bins, str) else list(time_bins)
        time_windows = self._normalize_time_windows(time_windows)
        requests = []

        for point_id, point in self._iter_id_point_geometries(id_point_geometries):
            point_id = self._format_point_id(point_id)
            for time_bin in time_bins:
                time_bin = str(time_bin)
                request_id = f"{point_id}{time_bin}{self.name}"
                requests.append(
                    (
                        request_id,
                        point_id,
                        time_bin,
                        point,
                        buffer,
                        self.ee_collection_id,
                        crs,
                        crs_transform,
                        dimensions,
                        Path(out_dir),
                        time_windows,
                    )
                )

        return requests

    @retry(tries=10, delay=1, backoff=2)
    def get_result(
        self,
        request_id: str,
        point_id: str,
        time_bin: str,
        point: dict[str, Any],
        buffer: int | float,
        ee_collection: str | ee.ImageCollection,
        crs: str,
        crs_transform: list[float],
        dimensions: str,
        out_dir: str | Path,
        time_windows: dict[str, tuple[str, str]] | None,
        ee_project: str | None = None,
        ee_opt_url: str | None = None,
    ) -> None:
        """Download one image patch from Earth Engine."""
        if self.initialize_ee:
            self._initialize_earth_engine(
                ee_project=ee_project,
                ee_opt_url=ee_opt_url,
            )

        source_name = self.name.lower()

        collection = self._collection_from_request(ee_collection)
        collection = self.apply_global_filter(collection)

        point_geometry = self._point_geometry(point)
        region = point_geometry.buffer(buffer).bounds()
        collection = collection.filterBounds(region)

        time_windows = self._normalize_time_windows(time_windows)
        window_collections = self.filter_time_windows(
            collection=collection,
            time_windows=time_windows,
            region=region,
        )
        images = [
            self.preprocess(
                collection=window_collection,
                window_name=window_name,
                time_bin=time_bin,
                region=region,
                crs=crs,
                crs_transform=crs_transform,
            )
            for window_name, window_collection in window_collections
        ]
        if not images:
            raise ValueError(f"No image outputs generated for request {request_id}")

        image = ee.Image.cat(images) if len(images) > 1 else images[0]
        self._download_image(
            image=image,
            region=region,
            dimensions=dimensions,
            out_dir=Path(out_dir),
            time_bin=time_bin,
            point_id=point_id,
            source_name=source_name,
        )

    def apply_global_filter(
        self,
        collection: ee.ImageCollection,
    ) -> ee.ImageCollection:
        """Apply filters configured for this source."""
        for global_filter in self.global_filter:
            collection = collection.filter(global_filter)
        return collection

    def filter_time_window(
        self,
        collection: ee.ImageCollection,
        window_name: str,
        start_date: str,
        end_date: str,
        region: ee.Geometry,
    ) -> ee.ImageCollection:
        """Filter a collection for one time window.

        Child classes can override this when a source needs temporal filtering
        that is not expressible as a simple ``filterDate`` call.
        """
        return collection.filterDate(start_date, end_date)

    def preprocess(
        self,
        collection: ee.ImageCollection,
        window_name: str | None,
        time_bin: str,
        region: ee.Geometry,
        crs: str,
        crs_transform: list[float],
    ) -> ee.Image:
        """Prepare one filtered collection as a single image for download.

        The default behavior selects the first image from the collection and
        reprojects it. Child classes can override this for compositing, masking,
        band selection, band renaming, mosaicking, or source-specific transforms.
        """
        return ee.Image(collection.first()).reproject(
            crs=crs,
            crsTransform=crs_transform,
        )

    def filter_time_windows(
        self,
        collection: ee.ImageCollection,
        time_windows: dict[str, tuple[str, str]],
        region: ee.Geometry,
    ):
        if not time_windows:
            yield None, collection
            return

        for window_name, (start_date, end_date) in time_windows.items():
            yield (
                window_name,
                self.filter_time_window(
                    collection=collection,
                    window_name=window_name,
                    start_date=start_date,
                    end_date=end_date,
                    region=region,
                ),
            )

    def _collection_from_request(
        self,
        ee_collection: str | ee.ImageCollection,
    ) -> ee.ImageCollection:
        if isinstance(ee_collection, str):
            return ee.ImageCollection(ee_collection)
        return ee_collection

    def _normalize_time_windows(
        self,
        time_windows: dict[str, tuple[str, str]] | None,
    ) -> dict[str, tuple[str, str]]:
        if time_windows is None:
            return {}

        normalized = {}
        for window_name, dates in time_windows.items():
            if len(dates) != 2:
                raise ValueError(
                    "time_windows must map each window name to "
                    "(start_date, end_date)"
                )
            start_date, end_date = dates
            normalized[str(window_name)] = (str(start_date), str(end_date))
        return normalized

    def _normalize_global_filter(
        self,
        global_filter: ee.Filter | list[ee.Filter] | None,
    ) -> list[ee.Filter]:
        if global_filter is None:
            return []
        if isinstance(global_filter, list):
            return global_filter
        return [global_filter]

    def _initialize_earth_engine(
        self,
        ee_project: str | None = None,
        ee_opt_url: str | None = None,
    ) -> None:
        kwargs = {}
        if ee_project is not None:
            kwargs["project"] = ee_project
        if ee_opt_url is not None:
            kwargs["opt_url"] = ee_opt_url
        ee.Initialize(**kwargs)

    def _iter_id_point_geometries(self, id_point_geometries: Iterable[Any]):
        for index, item in enumerate(id_point_geometries):
            if isinstance(item, tuple) and len(item) == 2:
                yield item
            elif isinstance(item, dict) and "id" in item:
                point = (
                    item.get("geometry")
                    or item.get("point")
                    or item.get(".geo")
                    or item.get("geo")
                )
                if point is None and "coordinates" in item:
                    point = item
                yield item["id"], point
            else:
                yield index, item

    def _format_point_id(self, point_id: Any) -> str:
        if isinstance(point_id, int):
            return str(point_id).zfill(3)
        return str(point_id)

    def _point_geometry(self, point: dict[str, Any]) -> ee.Geometry:
        if point is None:
            raise ValueError("Point geometry cannot be None")
        if isinstance(point, ee.Geometry):
            return point
        if not isinstance(point, dict):
            raise ValueError(f"Unsupported point geometry: {point}")
        if point.get("type") == "Feature" and point.get("geometry") is not None:
            return self._point_geometry(point["geometry"])
        if "coordinates" in point:
            return ee.Geometry.Point(point["coordinates"])
        raise ValueError(f"Unsupported point geometry: {point}")

    def _download_image(
        self,
        image: ee.Image,
        region: ee.Geometry,
        dimensions: str,
        out_dir: Path,
        time_bin: str,
        point_id: str,
        source_name: str,
    ) -> None:
        file_dir = out_dir / time_bin / source_name
        file_dir.mkdir(parents=True, exist_ok=True)

        filename_time_bin = time_bin.replace("/", "_")
        filename = file_dir / f"{point_id}_{filename_time_bin}_{source_name}.tif"
        if filename.exists():
            return

        url = image.getDownloadURL(
            {
                "region": region,
                "dimensions": dimensions,
                "format": "GEO_TIFF",
            }
        )
        response = http_requests.get(url, stream=True)
        if response.status_code != 200:
            response.raise_for_status()
        with open(filename, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
