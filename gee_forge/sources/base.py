from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import ee
import requests as http_requests
from retry import retry


#: Points projected per Earth Engine round trip in ``project_points``.
PROJECTION_CHUNK_SIZE = 500

#: Cache of projected point coordinates, keyed by output CRS and input points.
_PROJECTED_POINT_CACHE: dict[tuple[Any, ...], dict[tuple[float, float], list[float]]] = {}


def parse_dimensions(dimensions: str | int | Iterable[int]) -> tuple[int, int]:
    """Parse an Earth Engine ``dimensions`` value into ``(width, height)``.

    Args:
        dimensions: ``"{width}x{height}"``, a single integer, or an iterable of
            one or two integers.

    Returns:
        Chip width and height in pixels.
    """
    if isinstance(dimensions, bool):
        raise ValueError(f"Unsupported dimensions: {dimensions!r}")

    if isinstance(dimensions, int):
        values = [dimensions]
    elif isinstance(dimensions, str):
        text = dimensions.strip().lower()
        try:
            values = [int(part) for part in text.split("x")]
        except ValueError as error:
            raise ValueError(f"Unsupported dimensions: {dimensions!r}") from error
    else:
        try:
            values = [int(value) for value in dimensions]
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unsupported dimensions: {dimensions!r}") from error

    if len(values) == 1:
        values = values * 2
    if len(values) != 2 or any(value <= 0 for value in values):
        raise ValueError(f"Unsupported dimensions: {dimensions!r}")
    return values[0], values[1]


def decompose_transform(crs_transform: Iterable[float]) -> tuple[float, float, float, float]:
    """Split a row-major Earth Engine ``crsTransform`` into scale and origin.

    Args:
        crs_transform: Affine transform as
            ``[xScale, xShearing, xTranslation, yShearing, yScale, yTranslation]``,
            matching ``ee.Projection.transform`` and ``ee.Image.reproject``.

    Returns:
        ``(x_scale, y_scale, x_origin, y_origin)``. ``y_scale`` is normally
        negative, placing the origin at the top-left corner of the grid.

    Raises:
        ValueError: If the transform is not six numbers, has a zero scale, or
            is rotated or sheared. Snapping a chip to a rotated grid is
            ambiguous, so those transforms are rejected rather than silently
            mishandled.
    """
    values = [float(value) for value in crs_transform]
    if len(values) != 6:
        raise ValueError(
            f"crs_transform must contain 6 numbers, got {len(values)}: {values}"
        )

    x_scale, x_shear, x_origin, y_shear, y_scale, y_origin = values
    if x_shear or y_shear:
        raise ValueError(
            "Rotated or sheared crs_transform is not supported for pixel-aligned "
            f"chips: {values}"
        )
    if not x_scale or not y_scale:
        raise ValueError(f"crs_transform must have non-zero scales: {values}")
    return x_scale, y_scale, x_origin, y_origin


def pixel_aligned_transform(
    x: float,
    y: float,
    crs_transform: Iterable[float],
    width: int,
    height: int,
) -> list[float]:
    """Build a chip ``crsTransform`` centred on a point and snapped to the grid.

    The chip keeps the scale and the pixel lattice of ``crs_transform``, so chips
    from different points, dates, and sources share one grid and co-register
    exactly. The point lands inside the pixel at ``(width // 2, height // 2)``,
    which places it within half a pixel of the chip centre.

    Args:
        x: Point easting, in the units of the output CRS.
        y: Point northing, in the units of the output CRS.
        crs_transform: Reference grid transform that defines scale and lattice.
        width: Chip width in pixels.
        height: Chip height in pixels.

    Returns:
        Row-major affine transform for the chip's top-left corner.
    """
    x_scale, y_scale, x_origin, y_origin = decompose_transform(crs_transform)
    column = math.floor((x - x_origin) / x_scale)
    row = math.floor((y - y_origin) / y_scale)
    chip_x = x_origin + (column - width // 2) * x_scale
    chip_y = y_origin + (row - height // 2) * y_scale
    return [x_scale, 0.0, chip_x, 0.0, y_scale, chip_y]


def chip_region(
    crs: str,
    chip_transform: Iterable[float],
    width: int,
    height: int,
    margin: float = 0.0,
) -> ee.Geometry:
    """Build the chip footprint as a rectangle in the output CRS.

    Args:
        crs: Output coordinate reference system.
        chip_transform: Chip transform from ``pixel_aligned_transform``.
        width: Chip width in pixels.
        height: Chip height in pixels.
        margin: Extra margin, in CRS units, added on every side.

    Returns:
        Planar rectangle covering the chip, expanded by ``margin``.
    """
    x_scale, y_scale, x_origin, y_origin = decompose_transform(chip_transform)
    x_min, x_max = sorted((x_origin, x_origin + width * x_scale))
    y_min, y_max = sorted((y_origin, y_origin + height * y_scale))
    return ee.Geometry.Rectangle(
        coords=[x_min - margin, y_min - margin, x_max + margin, y_max + margin],
        proj=crs,
        geodesic=False,
    )


def project_points(
    coordinates: Iterable[Iterable[float]],
    crs: str,
    max_error: float = 0.001,
) -> list[list[float]]:
    """Project longitude/latitude pairs into ``crs`` using Earth Engine.

    Points are deduplicated and sent in chunks, so the cost is a small number of
    round trips regardless of how many points are requested. Results are cached,
    so several sources sharing one point set and CRS only pay for the first call.

    Requires an initialized Earth Engine session on the calling process.

    Args:
        coordinates: Iterable of ``(longitude, latitude)`` pairs in EPSG:4326.
        crs: Target coordinate reference system.
        max_error: Error margin, in meters, passed to ``ee.Geometry.transform``.

    Returns:
        Projected ``[x, y]`` pairs, in the order of ``coordinates``.
    """
    points = [(float(longitude), float(latitude)) for longitude, latitude in coordinates]
    if not points:
        return []

    cache_key = (crs, max_error, tuple(points))
    cached = _PROJECTED_POINT_CACHE.get(cache_key)
    if cached is None:
        unique = list(dict.fromkeys(points))
        cached = {}
        for start in range(0, len(unique), PROJECTION_CHUNK_SIZE):
            chunk = unique[start : start + PROJECTION_CHUNK_SIZE]
            projected = (
                ee.Geometry.MultiPoint([list(point) for point in chunk])
                .transform(crs, max_error)
                .coordinates()
                .getInfo()
            )
            if len(projected) != len(chunk):
                raise ValueError(
                    f"Earth Engine returned {len(projected)} projected points for "
                    f"{len(chunk)} inputs while projecting into {crs}"
                )
            cached.update(
                (point, [float(x), float(y)])
                for point, (x, y) in zip(chunk, projected)
            )
        _PROJECTED_POINT_CACHE[cache_key] = cached

    return [list(cached[point]) for point in points]


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
        crs: str,
        crs_transform: list[float],
        dimensions: str,
        time_bins: str | Iterable[str],
        time_windows: dict[str, tuple[str, str]] | None,
        out_dir: str | Path,
        filter_margin: int | float = 0.0,
    ) -> list[tuple]:
        """Generate download request tuples for this source.

        Each point gets its own ``crsTransform``, centred on the point and
        snapped to the lattice of ``crs_transform``. The chip extent is therefore
        defined entirely by ``crs_transform`` and ``dimensions``, and every chip
        lands on the same pixel grid.

        Requires an initialized Earth Engine session on the calling process,
        because points are projected into ``crs`` to compute their transforms.

        Args:
            id_point_geometries: Iterable containing point ids and GeoJSON-like
                point geometries. Supported items are ``(id, geometry)`` tuples,
                dicts with ``id`` plus ``geometry``/``point``/``.geo``, or raw
                point geometries, which are enumerated.
            crs: Output coordinate reference system.
            crs_transform: Reference grid transform, as
                ``[xScale, 0, xTranslation, 0, yScale, yTranslation]``. Only its
                scale and lattice are used; each chip gets its own origin.
            dimensions: Output chip dimensions as ``"{width}x{height}"``.
            time_bins: Output bin or bins used to organize downloaded files.
            time_windows: Mapping of ``window_name`` to ``(start_date, end_date)``.
            out_dir: Root directory where chips will be written.
            filter_margin: Extra margin, in CRS units, added to the chip
                footprint when filtering the source collection. Does not affect
                the output grid.

        Returns:
            Request tuples that can be consumed by ``get_result``.
        """
        time_bins = [time_bins] if isinstance(time_bins, str) else list(time_bins)
        time_windows = self._normalize_time_windows(time_windows)
        width, height = parse_dimensions(dimensions)

        id_points = list(self._iter_id_point_geometries(id_point_geometries))
        projected = project_points(
            (self._point_coordinates(point) for _, point in id_points),
            crs=crs,
        )

        requests = []
        for (point_id, point), (x, y) in zip(id_points, projected):
            point_id = self._format_point_id(point_id)
            chip_transform = pixel_aligned_transform(
                x=x,
                y=y,
                crs_transform=crs_transform,
                width=width,
                height=height,
            )
            for time_bin in time_bins:
                time_bin = str(time_bin)
                request_id = f"{point_id}{time_bin}{self.name}"
                requests.append(
                    (
                        request_id,
                        point_id,
                        time_bin,
                        point,
                        self.ee_collection_id,
                        crs,
                        chip_transform,
                        dimensions,
                        Path(out_dir),
                        time_windows,
                        filter_margin,
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
        ee_collection: str | ee.ImageCollection,
        crs: str,
        chip_transform: list[float],
        dimensions: str,
        out_dir: str | Path,
        time_windows: dict[str, tuple[str, str]] | None,
        filter_margin: int | float = 0.0,
        ee_project: str | None = None,
        ee_opt_url: str | None = None,
    ) -> None:
        """Download one image patch from Earth Engine.

        ``chip_transform`` is used both to reproject the image and to request the
        download, so the export is a plain window read on the grid the image is
        already pinned to, with no second reprojection.
        """
        if self.initialize_ee:
            self._initialize_earth_engine(
                ee_project=ee_project,
                ee_opt_url=ee_opt_url,
            )

        source_name = self.name.lower()
        width, height = parse_dimensions(dimensions)

        collection = self._collection_from_request(ee_collection)
        collection = self.apply_global_filter(collection)

        region = chip_region(
            crs=crs,
            chip_transform=chip_transform,
            width=width,
            height=height,
            margin=filter_margin,
        )
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
                crs_transform=chip_transform,
            )
            for window_name, window_collection in window_collections
        ]
        if not images:
            raise ValueError(f"No image outputs generated for request {request_id}")

        image = ee.Image.cat(images) if len(images) > 1 else images[0]
        self._download_image(
            image=image,
            crs=crs,
            chip_transform=chip_transform,
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

    def _point_coordinates(self, point: Any) -> tuple[float, float]:
        """Return a point's ``(longitude, latitude)`` on the client.

        ``ee.Geometry`` points are resolved with ``getInfo``, one call each, so
        plain GeoJSON points are preferred for large point sets.
        """
        if point is None:
            raise ValueError("Point geometry cannot be None")

        if isinstance(point, ee.Geometry):
            coordinates = point.getInfo().get("coordinates")
        elif isinstance(point, dict):
            if point.get("type") == "Feature" and point.get("geometry") is not None:
                return self._point_coordinates(point["geometry"])
            coordinates = point.get("coordinates")
        else:
            raise ValueError(f"Unsupported point geometry: {point}")

        if not isinstance(coordinates, (list, tuple)) or len(coordinates) < 2:
            raise ValueError(f"Unsupported point geometry: {point}")
        return float(coordinates[0]), float(coordinates[1])

    def _download_image(
        self,
        image: ee.Image,
        crs: str,
        chip_transform: list[float],
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

        # The output grid is requested explicitly. Passing a region instead would
        # let Earth Engine derive its own transform from the region bounds, which
        # resamples the image onto an unaligned grid and leaves seam artifacts.
        url = image.getDownloadURL(
            {
                "crs": crs,
                "crs_transform": list(chip_transform),
                "dimensions": dimensions,
                "format": "GEO_TIFF",
            }
        )
        response = http_requests.get(url, stream=True)
        if response.status_code != 200:
            response.raise_for_status()
        with open(filename, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
