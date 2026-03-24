from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr

from diffusion_models_enso.analysis import (
    build_enso_selection_dataset,
    build_monthly_diagnostics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_enso_composites.py"


def load_build_script():
    spec = importlib.util.spec_from_file_location("build_enso_composites", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_diffusion_dataset() -> xr.Dataset:
    months = np.array([12, 1])
    years = np.array([2030, 2031, 2032])
    samples = np.array([0, 1, 2])

    nino34_trefht = np.array(
        [
            [
                [-0.4, 0.1, 0.8],
                [-0.6, 0.0, 0.3],
                [0.2, -0.1, 0.4],
            ],
            [
                [0.5, -0.2, 0.1],
                [0.4, 0.0, -0.3],
                [np.nan, np.nan, np.nan],
            ],
        ]
    )
    global_trefht = nino34_trefht + 10.0
    file_name = np.array(
        [
            ["month12_2030.nc", "month12_2031.nc", "month12_2032.nc"],
            ["month01_2030.nc", "month01_2031.nc", ""],
        ],
        dtype="<U32",
    )

    return xr.Dataset(
        data_vars={
            "nino34_trefht": (("month", "year", "sample"), nino34_trefht),
            "global_trefht": (("month", "year", "sample"), global_trefht),
            "file_name": (("month", "year"), file_name),
        },
        coords={"month": months, "year": years, "sample": samples},
    )


def write_source_file(path: Path) -> None:
    samples = np.array([0, 1, 2])
    lat = np.array([-10.0, 0.0])
    lon = np.array([190.0, 220.0])
    values = np.array(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
            [[0.0, 2.0], [4.0, 6.0]],
        ]
    )
    xr.Dataset(
        data_vars={"TREFHT": (("samples", "lat", "lon"), values)},
        coords={"samples": samples, "lat": lat, "lon": lon},
    ).to_netcdf(path)


def write_source_files(directory: Path, *file_names: str) -> None:
    for file_name in file_names:
        write_source_file(directory / file_name)


def test_main_writes_enso_composites_dataset(tmp_path: Path) -> None:
    module = load_build_script()
    input_path = tmp_path / "enso_selection.nc"
    output_path = tmp_path / "enso_composites.nc"

    diagnostics = build_monthly_diagnostics(build_diffusion_dataset())
    diagnostics.attrs["source_input_dir"] = str(tmp_path)
    selection = build_enso_selection_dataset(diagnostics, months=[12])
    selection.to_netcdf(input_path)
    write_source_files(
        tmp_path,
        "month12_2030.nc",
        "month12_2031.nc",
        "month12_2032.nc",
    )

    module.main(
        input=input_path,
        variables=["TREFHT"],
        years=[2030, 2031],
        num_workers=1,
        output=output_path,
        overwrite=True,
    )

    composites = xr.open_dataset(output_path)
    try:
        assert "trefht_composite" in composites.data_vars
        assert composites["trefht_composite"].shape == (2, 2, 2, 2)
        assert composites["n_events"].sel(event_type="nino", year=2030).item() == 1
        assert composites.attrs["source_dataset_path"] == str(input_path)
    finally:
        composites.close()
