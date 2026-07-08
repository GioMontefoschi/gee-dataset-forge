from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path
from typing import Any, Iterable

from gee_forge.sources.base import ImageSource

EARTH_ENGINE_HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"


def _download_source_request(job: tuple[ImageSource, tuple, str | None, str | None]) -> None:
    source, request, ee_project, ee_opt_url = job
    source.get_result(*request, ee_project=ee_project, ee_opt_url=ee_opt_url)


class DownloadProgress:
    """Small stderr progress bar for source download jobs."""

    def __init__(self, total: int, width: int = 40):
        self.total = total
        self.width = width
        self.current = 0

    def update(self, step: int = 1) -> None:
        self.current += step
        self.render()

    def render(self) -> None:
        if self.total == 0:
            return

        ratio = self.current / self.total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        sys.stderr.write(
            f"\rDownloading [{bar}] {self.current}/{self.total} "
            f"({ratio:.0%})"
        )
        sys.stderr.flush()

    def close(self) -> None:
        if self.total > 0:
            sys.stderr.write("\n")
            sys.stderr.flush()


class ImageSourceDownloader:
    """Download image patches from a set of image sources."""

    def __init__(
        self,
        points: Any,
        image_sources: ImageSource | Iterable[ImageSource],
        ee_project: str | None = None,
        ee_opt_url: str | None = EARTH_ENGINE_HIGH_VOLUME_URL,
    ):
        self.points = points
        self.image_sources = self._normalize_image_sources(image_sources)
        self.ee_project = ee_project
        self.ee_opt_url = ee_opt_url
        self.requests: list[tuple[ImageSource, tuple, str | None, str | None]] = []

    def generate_requests(
        self,
        buffer: int | float,
        crs: str,
        crs_transform: list[float],
        dimensions: str,
        time_bins: str | Iterable[str],
        time_windows: dict[str, tuple[str, str]] | None,
        out_dir: str | Path,
    ) -> list[tuple[ImageSource, tuple, str | None, str | None]]:
        """Generate download jobs for all configured image sources."""
        points = self._resolve_points()
        jobs = []

        for source in self.image_sources:
            source_requests = source.get_requests(
                id_point_geometries=points,
                buffer=buffer,
                crs=crs,
                crs_transform=crs_transform,
                dimensions=dimensions,
                time_bins=time_bins,
                time_windows=time_windows,
                out_dir=out_dir,
            )
            jobs.extend(
                (source, request, self.ee_project, self.ee_opt_url)
                for request in source_requests
            )

        self.requests = jobs
        return jobs

    def bulk_download(
        self,
        out_dir: str | Path,
        buffer: int | float,
        crs: str,
        crs_transform: list[float],
        dimensions: str,
        time_bins: str | Iterable[str],
        time_windows: dict[str, tuple[str, str]] | None,
        num_workers: int = 20,
        show_progress: bool = True,
    ) -> list[tuple[ImageSource, tuple, str | None, str | None]]:
        """Generate and download all source requests."""
        jobs = self.generate_requests(
            buffer=buffer,
            crs=crs,
            crs_transform=crs_transform,
            dimensions=dimensions,
            time_bins=time_bins,
            time_windows=time_windows,
            out_dir=out_dir,
        )
        self.download_requests(
            jobs=jobs,
            num_workers=num_workers,
            show_progress=show_progress,
        )
        return jobs

    def download_requests(
        self,
        jobs: list[tuple[ImageSource, tuple, str | None, str | None]] | None = None,
        num_workers: int = 20,
        show_progress: bool = True,
    ) -> None:
        """Download previously generated jobs."""
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")

        jobs = self.requests if jobs is None else jobs
        progress = DownloadProgress(len(jobs)) if show_progress else None

        try:
            if num_workers == 1:
                for job in jobs:
                    _download_source_request(job)
                    if progress is not None:
                        progress.update()
                return

            with multiprocessing.Pool(num_workers) as pool:
                for _ in pool.imap_unordered(_download_source_request, jobs):
                    if progress is not None:
                        progress.update()
        finally:
            if progress is not None:
                progress.close()

    def _resolve_points(self):
        if hasattr(self.points, "aggregate_array"):
            return self.points.aggregate_array(".geo").getInfo()
        if isinstance(self.points, list):
            return self.points
        if isinstance(self.points, dict):
            return [self.points]
        if (
            isinstance(self.points, tuple)
            and len(self.points) == 2
            and isinstance(self.points[1], dict)
        ):
            return [self.points]
        return list(self.points)

    def _normalize_image_sources(
        self,
        image_sources: ImageSource | Iterable[ImageSource],
    ) -> list[ImageSource]:
        if isinstance(image_sources, ImageSource):
            return [image_sources]
        return list(image_sources)
