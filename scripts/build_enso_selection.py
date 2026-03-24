#!/usr/bin/env python3

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
import xarray as xr

from diffusion_models_enso.analysis import (
    DEFAULT_ENSO_MONTHS,
    DEFAULT_QUANTILE_NINA,
    DEFAULT_QUANTILE_NINO,
    build_enso_selection_dataset,
)
from diffusion_models_enso.utils import find_repo_root

LOGGER = logging.getLogger(__name__)


def main(
    input: Annotated[
        Path,
        typer.Option(help="Input monthly diagnostics NetCDF path."),
    ],
    months: Annotated[
        list[int],
        typer.Option(help="Month numbers to include. Defaults to DJF (12 1 2)."),
    ] = list(DEFAULT_ENSO_MONTHS),
    quantile_nino: Annotated[
        float,
        typer.Option(help="Upper quantile used to define El Nino events."),
    ] = DEFAULT_QUANTILE_NINO,
    quantile_nina: Annotated[
        float,
        typer.Option(help="Lower quantile used to define La Nina events."),
    ] = DEFAULT_QUANTILE_NINA,
    output: Annotated[
        Path | None,
        typer.Option(
            help="Output NetCDF path. Defaults to repo_root/data/enso_selection.nc."
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite an existing output file."),
    ] = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo_root = find_repo_root(Path(__file__))
    output_path = output or repo_root / "data" / "enso_selection.nc"

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Pass --overwrite to replace it."
        )

    LOGGER.info("Reading monthly diagnostics from %s", input)
    with xr.open_dataset(input) as monthly_diagnostics:
        enso_selection = build_enso_selection_dataset(
            monthly_diagnostics.load(),
            months=months,
            quantile_nino=quantile_nino,
            quantile_nina=quantile_nina,
            source_dataset_path=str(input),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing ENSO selection to %s", output_path)
    enso_selection.to_netcdf(output_path)
    LOGGER.info("Done")


if __name__ == "__main__":
    typer.run(main)
