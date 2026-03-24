#!/usr/bin/env python3

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
import xarray as xr

from diffusion_models_enso.analysis import (
    DEFAULT_COMPOSITE_VARIABLES,
    DEFAULT_COMPOSITE_YEARS,
    build_enso_composites_dataset,
)
from diffusion_models_enso.utils import find_repo_root

LOGGER = logging.getLogger(__name__)


def main(
    input: Annotated[
        Path,
        typer.Option(help="Input ENSO selection NetCDF path."),
    ],
    variables: Annotated[
        list[str],
        typer.Option(
            help="Source variables to composite. Defaults to TREFHT, PS, PRECT."
        ),
    ] = list(DEFAULT_COMPOSITE_VARIABLES),
    years: Annotated[
        list[int],
        typer.Option(
            help="Years to composite. Defaults to 2015, 2025, ..., 2085."
        ),
    ] = list(DEFAULT_COMPOSITE_YEARS),
    num_workers: Annotated[
        int | None,
        typer.Option(
            help="Number of worker processes. Parallelism is across variables."
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            help="Output NetCDF path. Defaults to repo_root/data/enso_composites.nc."
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite an existing output file."),
    ] = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    repo_root = find_repo_root(Path(__file__))
    output_path = output or repo_root / "data" / "enso_composites.nc"

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Pass --overwrite to replace it."
        )

    LOGGER.info("Reading ENSO selection from %s", input)
    with xr.open_dataset(input) as selection_dataset:
        # This script is intentionally thin: it loads the persisted ENSO event
        # table, delegates all composite building to the analysis layer, and
        # then writes one composite artifact back to disk.
        LOGGER.info("Variables: %s", tuple(variables))
        LOGGER.info("Years: %s", tuple(years))
        LOGGER.info(
            "Composite workers: %s",
            num_workers if num_workers is not None else "auto",
        )
        composites = build_enso_composites_dataset(
            selection_dataset.load(),
            variables=variables,
            years=years,
            source_dataset_path=str(input),
            num_workers=num_workers,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing ENSO composites to %s", output_path)
    composites.to_netcdf(output_path)
    LOGGER.info("Done")


if __name__ == "__main__":
    typer.run(main)
