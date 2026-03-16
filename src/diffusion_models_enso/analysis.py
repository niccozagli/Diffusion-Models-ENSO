from __future__ import annotations

from pathlib import Path

import numpy as np
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


def compute_nino34_index(
    diffusion_dataset: xr.Dataset,
    *,
    variable: str = "nino_34",
    sample_dim: str = "sample",
) -> xr.DataArray:
    """Center the stored Nino 3.4 temperatures by subtracting the ensemble mean."""
    nino34 = diffusion_dataset[variable]
    return nino34 - nino34.mean(dim=sample_dim, skipna=True)


def select_month(
    data_array: xr.DataArray,
    month: int,
    *,
    month_dim: str = "month",
    sample_dim: str = "sample",
    year_dim: str = "year",
    drop_empty: bool = True,
) -> xr.DataArray:
    """Select one month and optionally drop year rows with no valid samples."""
    month_data = data_array.sel({month_dim: month})
    if drop_empty:
        return drop_empty_years(
            month_data,
            sample_dim=sample_dim,
            year_dim=year_dim,
        )
    return month_data


def select_reference_samples(
    data_array: xr.DataArray,
    *,
    year: int | None = None,
    sample_dim: str = "sample",
    year_dim: str = "year",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return the max, min, and closest-to-zero samples for the selection."""
    selected = data_array
    if year is not None:
        selected = selected.sel({year_dim: year})

    max_sample = selected.idxmax(sample_dim)
    min_sample = selected.idxmin(sample_dim)
    neutral_sample = np.abs(selected).idxmin(sample_dim) #type:ignore

    return (
        selected.sel({sample_dim: max_sample}),
        selected.sel({sample_dim: min_sample}),
        selected.sel({sample_dim: neutral_sample}),
    )


def get_diffusion_source_file_name(
    diffusion_dataset: xr.Dataset,
    month: int,
    year: int,
    *,
    variable: str = "file_name",
    month_dim: str = "month",
    year_dim: str = "year",
) -> str:
    """Look up the original monthly NetCDF file recorded for a month/year pair."""
    return diffusion_dataset[variable].sel({month_dim: month, year_dim: year}).item()


def compute_sample_anomaly(
    source_dataset: xr.Dataset,
    sample: int,
    *,
    variable: str = "TREFHT",
    sample_dim: str = "samples",
) -> xr.DataArray:
    """Subtract the ensemble mean field from one sample in a source dataset."""
    field = source_dataset[variable].sel({sample_dim: sample})
    ensemble_mean = source_dataset[variable].mean(dim=sample_dim, skipna=True)
    return field - ensemble_mean


def reconstruct_sample_anomaly(
    diffusion_dataset: xr.Dataset,
    input_data_dir: Path,
    month: int,
    year: int,
    sample: int,
    *,
    variable: str = "TREFHT",
    file_name_var: str = "file_name",
    source_sample_dim: str = "samples",
) -> xr.DataArray:
    """Rebuild a sample anomaly field from diffusion metadata and source files."""
    file_name = get_diffusion_source_file_name(
        diffusion_dataset,
        month=month,
        year=year,
        variable=file_name_var,
    )
    source_path = Path(input_data_dir) / file_name
    with xr.open_dataset(source_path) as source_dataset:
        return compute_sample_anomaly(
            source_dataset,
            sample=sample,
            variable=variable,
            sample_dim=source_sample_dim,
        ).load()
