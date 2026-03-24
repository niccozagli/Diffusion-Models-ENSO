from __future__ import annotations

import xarray as xr


def drop_empty_years(
    data_array: xr.DataArray,
    *,
    sample_dim: str = "sample",
    year_dim: str = "year",
) -> xr.DataArray:
    """Remove year entries whose values are all NaN across the sample dimension."""
    valid_years = data_array.notnull().any(dim=sample_dim)
    return data_array.sel({year_dim: valid_years})
