#!/usr/bin/env python3

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import xarray as xr
from tqdm import tqdm

from diffusion_models_enso.utils import find_repo_root

DEFAULT_MONTHS = (12, 1, 2, 6, 7, 8)
LOGGER = logging.getLogger(__name__)


def build_month_arrays(
    files: list[Path], month: int, n_samples: int | None
) -> tuple[np.ndarray, np.ndarray, list[str], int, int]:
    if not files:
        raise FileNotFoundError(f"No files found for month {month:02}")

    with xr.open_dataset(files[0]) as first_ds:
        month_n_samples = first_ds.sizes["samples"]

    if n_samples is not None and month_n_samples != n_samples:
        raise ValueError(
            f"Month {month:02} has {month_n_samples} samples, expected {n_samples}"
        )

    month_n_years = len(files)

    nino34 = np.empty((month_n_years, month_n_samples))
    global_temp = np.empty((month_n_years, month_n_samples))

    for index_year, fp in enumerate(
        tqdm(files, desc=f"Processing month {month:02}", leave=False)
    ):
        with xr.open_dataset(fp) as ds:
            weights = xr.DataArray(
                np.cos(np.deg2rad(ds["lat"].to_numpy())),
                coords={"lat": ds["lat"]},
                dims=("lat",),
            )
            global_samples = (
                ds["TREFHT"].weighted(weights=weights).mean(dim=("lat", "lon"))
            )

            nino_region = ds["TREFHT"].sel(lat=slice(-5, 5), lon=slice(190, 240))
            nino_weights = xr.DataArray(
                np.cos(np.deg2rad(nino_region["lat"].to_numpy())),
                coords={"lat": nino_region["lat"]},
                dims=("lat",),
            )
            nino_samples = nino_region.weighted(weights=nino_weights).mean(
                dim=("lat", "lon")
            )

            if ds.sizes["samples"] != month_n_samples:
                raise ValueError(
                    f"File {fp.name} has {ds.sizes['samples']} samples, expected {month_n_samples}"
                )

            nino34[index_year, :] = nino_samples.values
            global_temp[index_year, :] = global_samples.values

    return (
        global_temp,
        nino34,
        [fp.name for fp in files],
        month_n_years,
        month_n_samples,
    )


def main(
    data_dir: Annotated[
        Path,
        typer.Option(help="Directory containing the monthly diffusion NetCDF files."),
    ],
    months: Annotated[
        list[int],
        typer.Option(help="Month numbers to process. Defaults to DJF and JJA."),
    ] = list(DEFAULT_MONTHS),
    start_year: Annotated[
        int,
        typer.Option(help="Year assigned to the first file after sorting."),
    ] = 2015,
    output: Annotated[
        Path | None,
        typer.Option(
            help="Output NetCDF path. Defaults to repo_root/data/diffusion.nc."
        ),
    ] = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    selected_months = tuple(months)
    repo_root = find_repo_root(Path(__file__))
    LOGGER.info("Using data directory %s", data_dir)
    LOGGER.info("Processing months %s", selected_months)
    n_samples: int | None = None

    global_by_month: list[np.ndarray] = []
    nino_by_month: list[np.ndarray] = []
    file_names_by_month: list[list[str]] = []
    year_counts: list[int] = []

    for month in selected_months:
        file_pattern = f"samples_governance_indexes_3944_month{month:02}*.nc"
        files = sorted(data_dir.glob(file_pattern))
        LOGGER.info("Month %02d: found %d files", month, len(files))

        global_temp, nino34, file_names, month_n_years, month_n_samples = (
            build_month_arrays(files, month, n_samples)
        )

        if n_samples is None:
            n_samples = month_n_samples

        global_by_month.append(global_temp)
        nino_by_month.append(nino34)
        file_names_by_month.append(file_names)
        year_counts.append(month_n_years)

    if n_samples is None:
        raise RuntimeError("No data was processed")

    n_years = max(year_counts)
    years = [start_year + i for i in range(n_years)]
    global_array = np.full((len(selected_months), n_years, n_samples), np.nan)
    nino_array = np.full((len(selected_months), n_years, n_samples), np.nan)
    max_name_len = max(len(name) for names in file_names_by_month for name in names)
    file_name_array = np.full(
        (len(selected_months), n_years), "", dtype=f"<U{max_name_len}"
    )

    for month_index, month_n_years in enumerate(year_counts):
        global_array[month_index, :month_n_years, :] = global_by_month[month_index]
        nino_array[month_index, :month_n_years, :] = nino_by_month[month_index]
        file_name_array[month_index, :month_n_years] = file_names_by_month[month_index]

    LOGGER.info(
        "Assembled arrays with shape month=%d year=%d sample=%d",
        len(selected_months),
        n_years,
        n_samples,
    )

    data_vars = {
        "global_trefht": (("month", "year", "sample"), global_array),
        "nino_34": (("month", "year", "sample"), nino_array),
        "file_name": (("month", "year"), file_name_array),
    }

    coords = {
        "month": np.asarray(selected_months),
        "sample": np.arange(n_samples),
        "year": years,
    }

    attrs = {
        "description": "Diffusion diagnostic for global and nino34 indices across DJF and JJA months",
        "nino34_lat_range": "-5 to 5",
        "nino34_lon_range": "190 to 240",
    }

    diffusion = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

    output_path = output or repo_root / "data" / "diffusion.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing NetCDF to %s", output_path)
    diffusion.to_netcdf(output_path)
    LOGGER.info("Done")


if __name__ == "__main__":
    typer.run(main)
