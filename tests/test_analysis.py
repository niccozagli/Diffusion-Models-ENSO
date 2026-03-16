from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from diffusion_models_enso.analysis import (
    compute_nino34_index,
    compute_sample_anomaly,
    drop_empty_years,
    get_diffusion_source_file_name,
    reconstruct_sample_anomaly,
    select_month,
    select_reference_samples,
)


def build_diffusion_dataset() -> xr.Dataset:
    months = np.array([12, 1])
    years = np.array([2030, 2031, 2032])
    samples = np.array([0, 1, 2])

    nino34 = np.array(
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
    global_trefht = nino34 + 10.0
    file_name = np.array(
        [
            ["month12_2030.nc", "month12_2031.nc", "month12_2032.nc"],
            ["month01_2030.nc", "month01_2031.nc", ""],
        ],
        dtype="<U32",
    )

    return xr.Dataset(
        data_vars={
            "nino_34": (("month", "year", "sample"), nino34),
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


def test_drop_empty_years_removes_all_nan_rows() -> None:
    diffusion = build_diffusion_dataset()

    january = diffusion["nino_34"].sel(month=1)
    trimmed = drop_empty_years(january)

    assert trimmed["year"].values.tolist() == [2030, 2031]


def test_compute_nino34_index_centers_each_month_year_group() -> None:
    diffusion = build_diffusion_dataset()

    nino34_index = compute_nino34_index(diffusion)
    centered = nino34_index.mean(dim="sample", skipna=True)

    np.testing.assert_allclose(
        centered.sel(month=12).values,
        np.zeros(3),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        centered.sel(month=1).sel(year=[2030, 2031]).values,
        np.zeros(2),
        atol=1e-12,
    )


def test_select_month_applies_drop_empty_years_by_default() -> None:
    diffusion = build_diffusion_dataset()

    january = select_month(diffusion["nino_34"], 1)

    assert january["year"].values.tolist() == [2030, 2031]


def test_select_reference_samples_returns_max_min_and_neutral_points() -> None:
    diffusion = build_diffusion_dataset()
    december = select_month(compute_nino34_index(diffusion), 12)

    max_value, min_value, neutral_value = select_reference_samples(december, year=2030)

    assert max_value["sample"].item() == 2
    assert min_value["sample"].item() == 0
    assert neutral_value["sample"].item() == 1
    assert max_value["year"].item() == 2030
    assert neutral_value.item() == 0.0


def test_get_diffusion_source_file_name_returns_expected_source_name() -> None:
    diffusion = build_diffusion_dataset()

    file_name = get_diffusion_source_file_name(diffusion, month=12, year=2031)

    assert file_name == "month12_2031.nc"


def test_compute_sample_anomaly_returns_field_minus_ensemble_mean() -> None:
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
    source_dataset = xr.Dataset(
        data_vars={"TREFHT": (("samples", "lat", "lon"), values)},
        coords={"samples": samples, "lat": lat, "lon": lon},
    )

    anomaly = compute_sample_anomaly(source_dataset, sample=1)

    expected = np.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(anomaly.values, expected)


def test_reconstruct_sample_anomaly_reads_original_file(tmp_path: Path) -> None:
    diffusion = build_diffusion_dataset()
    source_file = tmp_path / "month12_2030.nc"
    write_source_file(source_file)

    december = select_month(compute_nino34_index(diffusion), 12)
    _, _, neutral_value = select_reference_samples(december, year=2030)

    anomaly = reconstruct_sample_anomaly(
        diffusion,
        input_data_dir=tmp_path,
        month=12,
        year=neutral_value["year"].item(),
        sample=neutral_value["sample"].item(),
    )

    expected = np.array([[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(anomaly.values, expected)
    assert anomaly["lat"].values.tolist() == [-10.0, 0.0]
