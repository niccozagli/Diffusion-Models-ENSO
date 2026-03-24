from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr

from diffusion_models_enso.analysis import build_monthly_diagnostics

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_enso_selection.py"


def load_build_script():
    spec = importlib.util.spec_from_file_location("build_enso_selection", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_monthly_diagnostics_dataset() -> xr.Dataset:
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

    month_index = xr.Dataset(
        data_vars={
            "nino34_trefht": (("month", "year", "sample"), nino34_trefht),
            "global_trefht": (("month", "year", "sample"), global_trefht),
            "file_name": (("month", "year"), file_name),
        },
        coords={"month": months, "year": years, "sample": samples},
    )
    return build_monthly_diagnostics(month_index)


def test_main_writes_enso_selection_dataset(tmp_path: Path) -> None:
    module = load_build_script()
    input_path = tmp_path / "monthly_diagnostics.nc"
    output_path = tmp_path / "enso_selection.nc"
    diagnostics = build_monthly_diagnostics_dataset()
    diagnostics.to_netcdf(input_path)

    module.main(
        input=input_path,
        months=[12],
        output=output_path,
        overwrite=True,
    )

    selection = xr.open_dataset(output_path)
    try:
        assert selection.sizes["event"] == 4
        assert set(selection["event_type"].values.tolist()) == {"nino", "nina"}
        assert selection.attrs["source_dataset_path"] == str(input_path)
        assert selection.attrs["threshold_nino"] > selection.attrs["threshold_nina"]
        assert all(
            file_name.startswith("month12_")
            for file_name in selection["file_name"].values.tolist()
        )
    finally:
        selection.close()
