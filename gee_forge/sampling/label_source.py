from __future__ import annotations

from collections.abc import Callable, Sequence

import ee

from gee_forge.sources.cdl import remap_crops


Number = int | float
ImageFactory = Callable[[ee.Geometry], ee.Image]
ImageRemapper = Callable[[ee.Image], ee.Image]


class LabelSource:
    """Sampling-label configuration for Earth Engine label images.

    A label source owns the label image construction and the metadata
    MontefSampling needs to draw and validate binary samples.
    """

    def __init__(
        self,
        image: ee.Image | ImageFactory,
        *,
        band_name: str = "label",
        sample_value: Number = 1,
        other_value: Number = 0,
        sample_scale: Number = 30,
        projection: ee.Projection | None = None,
        remap_from_values: Sequence[Number] | None = None,
        remap_to_values: Sequence[Number] | None = None,
        remap_band_name: str | None = None,
        remap_default_value: Number = 0,
        remap_function: ImageRemapper | None = None,
        clip_to_roi: bool = True,
    ):
        """Initialize a generic label source.

        Args:
            image: Label image or factory that receives the ROI and returns a
                label image.
            band_name: Output band used for sampling.
            sample_value: Binary value to sample and count inside chip
                footprints.
            other_value: Binary value explicitly skipped by stratified sampling.
            sample_scale: Scale passed to Earth Engine stratified sampling.
            projection: Optional projection used for grid construction and label
                reprojection. When omitted, the sampling layer projection is
                used.
            remap_from_values: Source class values to remap.
            remap_to_values: Target binary values corresponding to
                ``remap_from_values``.
            remap_band_name: Source band to remap. Required when the source
                image has multiple bands.
            remap_default_value: Value assigned by Earth Engine remap for
                classes not listed in ``remap_from_values``.
            remap_function: Optional custom remapper for sources whose class
                logic is easier to express as code.
            clip_to_roi: Whether to clip the final sampling layer to the ROI.
        """
        if (remap_from_values is None) != (remap_to_values is None):
            raise ValueError("remap_from_values and remap_to_values must be paired")
        if remap_from_values is not None and remap_function is not None:
            raise ValueError("Use either remap values or remap_function, not both")
        if (
            remap_from_values is not None
            and len(remap_from_values) != len(remap_to_values or [])
        ):
            raise ValueError("remap_from_values and remap_to_values differ in length")

        self.image = image
        self.band_name = band_name
        self.sample_value = sample_value
        self.other_value = other_value
        self.sample_scale = sample_scale
        self.projection = projection
        self.remap_from_values = remap_from_values
        self.remap_to_values = remap_to_values
        self.remap_band_name = remap_band_name
        self.remap_default_value = remap_default_value
        self.remap_function = remap_function
        self.clip_to_roi = clip_to_roi

    def sampling_layer(self, roi: ee.Geometry) -> ee.Image:
        """Build the binary sampling image for an ROI."""
        image = self._image_for(roi)
        if self.remap_function is not None:
            layer = self.remap_function(image)
        elif self.remap_from_values is not None:
            layer = (
                image.remap(
                    self.remap_from_values,
                    self.remap_to_values,
                    defaultValue=self.remap_default_value,
                    bandName=self.remap_band_name,
                )
                .select("remapped")
                .rename(self.band_name)
            )
        else:
            layer = image.select(self.band_name)

        if self.clip_to_roi:
            layer = layer.clip(roi)
        if self.projection is not None:
            layer = layer.reproject(self.projection)
        return layer

    def projection_for(self, roi: ee.Geometry) -> ee.Projection:
        """Return the projection MontefSampling should use for its grid."""
        if self.projection is not None:
            return self.projection
        return self.sampling_layer(roi).projection()

    def _image_for(self, roi: ee.Geometry) -> ee.Image:
        if callable(self.image):
            return ee.Image(self.image(roi))
        return ee.Image(self.image)


class CDLBinaryLabelSource(LabelSource):
    """USDA CDL label source remapped to cultivated/non-cultivated classes."""

    def __init__(self, year: int):
        self.year = year
        super().__init__(
            image=self._image_for_year,
            band_name="label",
            sample_value=2,
            other_value=1,
            sample_scale=30,
            remap_function=remap_crops,
        )

    def _image_for_year(self, roi: ee.Geometry) -> ee.Image:
        start_date = f"{self.year}-01-01"
        end_date = f"{self.year + 1}-01-01"
        return ee.Image(
            ee.ImageCollection("USDA/NASS/CDL")
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .first()
        )
