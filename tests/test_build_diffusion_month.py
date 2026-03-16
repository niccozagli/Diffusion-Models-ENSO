from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from diffusion_models_enso.utils import find_repo_root

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_diffusion_month.py"


def load_build_script():
    spec = importlib.util.spec_from_file_location("build_diffusion_month", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_month_file(path: Path, sample_values: list[float]) -> None:
    lat = np.array([-5.0, 0.0, 5.0])
    lon = np.array([190.0, 220.0])
    samples = np.arange(len(sample_values))
    data = np.asarray(sample_values, dtype=float)[:, None, None] * np.ones(
        (len(sample_values), len(lat), len(lon)),
        dtype=float,
    )
    ds = xr.Dataset(
        data_vars={"TREFHT": (("samples", "lat", "lon"), data)},
        coords={"samples": samples, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)


def test_find_repo_root_from_nested_path() -> None:
    nested_file = REPO_ROOT / "src" / "diffusion_models_enso" / "utils" / "paths.py"
    assert find_repo_root(nested_file) == REPO_ROOT


def test_build_month_arrays_returns_expected_values(tmp_path: Path) -> None:
    module = load_build_script()
    file_one = tmp_path / "samples_governance_indexes_3944_month12_a.nc"
    file_two = tmp_path / "samples_governance_indexes_3944_month12_b.nc"
    write_month_file(file_one, [1.0, 2.0, 3.0])
    write_month_file(file_two, [4.0, 5.0, 6.0])

    global_temp, nino34, file_names, n_years, n_samples = module.build_month_arrays(
        [file_one, file_two],
        month=12,
        n_samples=None,
    )

    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_allclose(global_temp, expected)
    np.testing.assert_allclose(nino34, expected)
    assert file_names == [file_one.name, file_two.name]
    assert n_years == 2
    assert n_samples == 3


def test_build_month_arrays_rejects_inconsistent_sample_count(tmp_path: Path) -> None:
    module = load_build_script()
    file_one = tmp_path / "samples_governance_indexes_3944_month01_a.nc"
    file_two = tmp_path / "samples_governance_indexes_3944_month01_b.nc"
    write_month_file(file_one, [1.0, 2.0, 3.0])
    write_month_file(file_two, [4.0, 5.0])

    with pytest.raises(ValueError, match="expected 3"):
        module.build_month_arrays([file_one, file_two], month=1, n_samples=None)
