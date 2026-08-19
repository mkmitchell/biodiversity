"""Shared GEE covariate helpers for point sampling and raster export."""

from __future__ import annotations

import ee

GMTED2010_FULL = "USGS/GMTED2010_FULL"
GMTED_STD_BAND = "gmted_std_500m"


def compute_gmted_std_500m(radius_m: int = 500) -> ee.Image:
    """
    Neighborhood mean of GMTED2010 elevation standard deviation within radius_m.

    Uses the ``std`` band from USGS/GMTED2010_FULL (global DEM roughness at
  ~232 m). Focal mean at 500 m matches the Dynamic World 500 m buffer scale.
    """
    gmted_std = ee.Image(GMTED2010_FULL).select("std")
    kernel = ee.Kernel.circle(radius_m, "meters", normalize=True)
    return (
        gmted_std.reduceNeighborhood(ee.Reducer.mean(), kernel)
        .rename(GMTED_STD_BAND)
        .toFloat()
    )
