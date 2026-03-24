#!/usr/bin/env python3

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from diffusion_models_enso.analysis import (
    DEFAULT_MONTHS,
    DEFAULT_SOURCE_FILE_PATTERN,
    DEFAULT_SOURCE_VARIABLE,
    build_monthly_diagnostics_dataset,
)
from diffusion_models_enso.utils import find_repo_root

LOGGER = logging.getLogger(__name__)


def main(
    input_dir: Annotated[
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
    pattern: Annotated[
        str,
        typer.Option(help="Filename pattern template used to discover month files."),
    ] = DEFAULT_SOURCE_FILE_PATTERN,
    variable: Annotated[
        str,
        typer.Option(help="Variable read from the source NetCDF files."),
    ] = DEFAULT_SOURCE_VARIABLE,
    output: Annotated[
        Path | None,
        typer.Option(
            help="Output NetCDF path. Defaults to repo_root/data/monthly_diagnostics.nc."
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite an existing output file."),
    ] = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo_root = find_repo_root(Path(__file__))
    output_path = output or repo_root / "data" / "monthly_diagnostics.nc"

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Pass --overwrite to replace it."
        )

    LOGGER.info("Using data directory %s", input_dir)
    LOGGER.info("Processing months %s", tuple(months))
    diagnostics = build_monthly_diagnostics_dataset(
        input_dir=input_dir,
        months=months,
        start_year=start_year,
        variable=variable,
        file_pattern=pattern,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing NetCDF to %s", output_path)
    diagnostics.to_netcdf(output_path)
    LOGGER.info("Done")


if __name__ == "__main__":
    typer.run(main)
