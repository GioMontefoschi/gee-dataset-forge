from __future__ import annotations

import ee

from gee_forge.sampling.label_source import LabelSource


class MontefSampling:
    """Grid-based Earth Engine sampler for binary maps.

    The sampler tiles a region of interest into projected grid cells, keeps only
    cells with valid data at each corner, draws one point from the requested
    binary-map value per valid cell, and filters candidate points by that value's
    pixel coverage in the final chip footprint.
    """

    _ACCEPTED_PROPERTY = "_montef_accepted"

    def __init__(
        self,
        roi: ee.Geometry,
        label_source: LabelSource,
        grid_size: int = 8000,
        buffer: int | float = 2240,
        min_sample_fraction: float = 0.1,
        seed: int = 42,
        grid_buffer_padding: int | float = 10,
    ):
        """Initialize the sampling algorithm.

        Args:
            roi: Region of interest where samples may be generated.
            label_source: Label-source configuration used to build the binary
                sampling layer, grid projection, class band, class values, and
                stratified-sampling scale.
            grid_size: Side length, in meters, of each grid cell.
            buffer: Chip side length in meters.
            min_sample_fraction: Minimum fraction of ``sample_value`` pixels
                required inside the candidate chip footprint.
            seed: Random seed passed to Earth Engine stratified sampling.
            grid_buffer_padding: Extra meters added before shrinking grid cells
                so chip footprints stay inside their cells.
        """
        self.roi = roi
        self.label_source = label_source
        self.sampling_layer = label_source.sampling_layer(roi)
        self.grid_proj = label_source.projection or self.sampling_layer.projection()
        self.grid_size = grid_size
        self.buffer = buffer
        self.min_sample_fraction = min_sample_fraction
        self.seed = seed
        self.band_name = label_source.band_name
        self.sample_value = label_source.sample_value
        self.other_value = label_source.other_value
        self.sample_scale = label_source.sample_scale
        self.grid_buffer_padding = grid_buffer_padding

    def create_grid(
        self,
        buffer: int | float | None = None,
        grid_size: int | None = None,
    ) -> ee.FeatureCollection:
        """Create valid grid cells for spatially distributed sampling."""
        buffer = self.buffer if buffer is None else buffer
        grid_size = self.grid_size if grid_size is None else grid_size
        shrink_distance = buffer / 2

        roi = self.roi.transform(self.grid_proj, maxError=10)
        grid = roi.coveringGrid(self.grid_proj, grid_size)
        grid = grid.map(lambda f: f.setGeometry(f.geometry().buffer(-shrink_distance)))

        def valid_cell(feature):
            verts = ee.List(feature.geometry().coordinates().get(0)).slice(0, 4)
            points = ee.FeatureCollection(
                verts.map(lambda coords: ee.Feature(ee.Geometry.Point(coords)))
            )
            samples = self.sampling_layer.sampleRegions(collection=points)
            values = samples.aggregate_array(self.band_name)
            valid = ee.Algorithms.IsEqual(ee.Number(ee.List(values).size()), 4)
            return feature.set("valid", valid)

        return grid.map(valid_cell).filter(ee.Filter.eq("valid", True))

    def sample(self) -> ee.FeatureCollection:
        """Run the sampling algorithm and return accepted sample points."""
        grid = self.create_grid(
            buffer=self.buffer + self.grid_buffer_padding,
            grid_size=self.grid_size,
        )
        scale = self.sampling_layer.projection().nominalScale().getInfo()
        min_sample_pixels = int(
            self.min_sample_fraction * ((self.buffer / scale) ** 2)
        )
        footprint_buffer = self.buffer / 2

        def sample_points(feature):
            return self.sampling_layer.stratifiedSample(
                numPoints=1,
                region=feature.geometry(),
                scale=self.sample_scale,
                classBand=self.band_name,
                classValues=[self.other_value, self.sample_value],
                classPoints=[0, 1],
                geometries=True,
                seed=self.seed,
            )

        points = grid.map(sample_points).flatten()

        def accept_point(feature):
            region = feature.geometry().buffer(footprint_buffer).bounds()
            sample_count = (
                self.sampling_layer.updateMask(
                    self.sampling_layer.eq(self.sample_value)
                )
                .reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=region,
                    scale=scale,
                    maxPixels=1e9,
                )
            )
            accepted = ee.Algorithms.If(
                ee.Number(sample_count.get(self.band_name)).gte(
                    ee.Number(min_sample_pixels)
                ),
                True,
                False,
            )
            return feature.set(self._ACCEPTED_PROPERTY, accepted)

        points = points.map(accept_point)
        return points.filter(ee.Filter.eq(self._ACCEPTED_PROPERTY, True)).map(
            lambda feature: feature.select([self.band_name])
        )

    def run(self) -> ee.FeatureCollection:
        """Alias for ``sample`` for callers that prefer algorithm-style naming."""
        return self.sample()
