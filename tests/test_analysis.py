from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from diffusion_models_enso.analysis import (
    build_enso_composites_dataset,
    build_selected_event_composite,
    build_enso_selection_dataset,
    build_event_anomaly_matrix,
    build_monthly_diagnostics,
    build_monthly_diagnostics_dataset,
    compute_nino34_index,
    compute_sample_anomaly,
    get_diffusion_source_file_name,
    prepare_selected_event_anomaly,
    reconstruct_sample_anomaly,
    reconstruct_selected_event_anomaly,
    select_month,
    select_reference_samples,
)
from diffusion_models_enso.utils import drop_empty_years


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
            "nino34_trefht": (("month", "year", "sample"), nino34),
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


def test_drop_empty_years_removes_all_nan_rows() -> None:
    diffusion = build_diffusion_dataset()

    january = diffusion["nino34_trefht"].sel(month=1)
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


def test_build_monthly_diagnostics_relabels_months_and_adds_index() -> None:
    month_index = build_diffusion_dataset()
    month_index.attrs["source_artifact"] = "month_index.nc"

    diagnostics = build_monthly_diagnostics(month_index)

    assert diagnostics["month"].values.tolist() == ["December", "January"]
    assert diagnostics["month_number"].values.tolist() == [12, 1]
    assert diagnostics.attrs["source_artifact"] == "month_index.nc"
    assert "nino34_index_definition" in diagnostics.attrs

    centered = diagnostics["nino34_index"].mean(dim="sample", skipna=True)
    np.testing.assert_allclose(
        centered.sel(month="December").values,
        np.zeros(3),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        centered.sel(month="January").sel(year=[2030, 2031]).values,
        np.zeros(2),
        atol=1e-12,
    )


def test_select_month_applies_drop_empty_years_by_default() -> None:
    diffusion = build_diffusion_dataset()

    january = select_month(diffusion["nino34_trefht"], 1)

    assert january["year"].values.tolist() == [2030, 2031]


def test_select_month_supports_numeric_lookup_on_labeled_months() -> None:
    diagnostics = build_monthly_diagnostics(build_diffusion_dataset())

    january = select_month(diagnostics["nino34_index"], 1)

    assert january["year"].values.tolist() == [2030, 2031]
    assert january["month"].item() == "January"


def test_build_monthly_diagnostics_dataset_builds_new_schema_from_raw_files(
    tmp_path: Path,
) -> None:
    write_source_files(
        tmp_path,
        "samples_governance_indexes_3944_month12_a.nc",
        "samples_governance_indexes_3944_month12_b.nc",
        "samples_governance_indexes_3944_month01_a.nc",
    )

    diagnostics = build_monthly_diagnostics_dataset(
        input_dir=tmp_path,
        months=[12, 1],
        start_year=2030,
    )

    assert diagnostics["month"].values.tolist() == ["December", "January"]
    assert diagnostics["month_number"].values.tolist() == [12, 1]
    assert "nino34_trefht" in diagnostics.data_vars
    assert "nino34_index" in diagnostics.data_vars
    assert (
        diagnostics["file_name"]
        .sel(month="December", year=2030)
        .item()
        .endswith("month12_a.nc")
    )
    assert diagnostics["file_name"].sel(month="January", year=2031).item() == ""


def test_select_reference_samples_returns_max_min_and_neutral_points() -> None:
    diffusion = build_diffusion_dataset()
    december = select_month(compute_nino34_index(diffusion), 12)

    max_value, min_value, neutral_value = select_reference_samples(december, year=2030)

    assert max_value["sample"].item() == 2
    assert min_value["sample"].item() == 0
    assert neutral_value["sample"].item() == 1
    assert max_value["year"].item() == 2030
    assert np.isclose(neutral_value.item(), -0.06666666666666665)


def test_build_enso_selection_dataset_carries_event_file_links() -> None:
    diagnostics = build_monthly_diagnostics(build_diffusion_dataset())
    diagnostics.attrs["source_input_dir"] = "/tmp/raw"

    selection = build_enso_selection_dataset(diagnostics, months=[12])

    assert selection.sizes["event"] == 4
    assert set(selection["event_type"].values.tolist()) == {"nino", "nina"}
    assert selection["month"].values.tolist() == ["December"] * 4
    assert selection["month_number"].values.tolist() == [12] * 4
    assert all(
        file_name.startswith("month12_")
        for file_name in selection["file_name"].values.tolist()
    )
    assert selection.attrs["pipeline_step"] == "enso_selection"
    assert selection.attrs["selection_scope"] == "shared_threshold_across_months"
    assert selection.attrs["threshold_nino"] > selection.attrs["threshold_nina"]
    assert selection.attrs["source_dataset_id"] == diagnostics.attrs["dataset_id"]
    assert selection.attrs["source_input_dir"] == "/tmp/raw"


def test_reconstruct_selected_event_anomaly_uses_selection_file_name(
    tmp_path: Path,
) -> None:
    diagnostics = build_monthly_diagnostics(build_diffusion_dataset())
    diagnostics.attrs["source_input_dir"] = str(tmp_path)
    selection = build_enso_selection_dataset(diagnostics, months=[12])
    write_source_files(
        tmp_path,
        "month12_2030.nc",
        "month12_2031.nc",
        "month12_2032.nc",
    )

    anomaly = reconstruct_selected_event_anomaly(
        selection,
        event=0,
    )

    expected_by_sample = {
        0: np.zeros((2, 2)),
        1: np.ones((2, 2)),
        2: -np.ones((2, 2)),
    }
    sample = selection["sample"].sel(event=0).item()
    np.testing.assert_allclose(anomaly.values, expected_by_sample[sample])


def test_prepare_selected_event_anomaly_uses_one_selected_event(tmp_path: Path) -> None:
    diagnostics = build_monthly_diagnostics(build_diffusion_dataset())
    diagnostics.attrs["source_input_dir"] = str(tmp_path)
    selection = build_enso_selection_dataset(diagnostics, months=[12])
    selected_event = selection.sel(event=0)
    write_source_files(
        tmp_path,
        "month12_2030.nc",
        "month12_2031.nc",
        "month12_2032.nc",
    )

    anomaly = prepare_selected_event_anomaly(
        selected_event,
    )

    expected_by_sample = {
        0: np.zeros((2, 2)),
        1: np.ones((2, 2)),
        2: -np.ones((2, 2)),
    }
    sample = selected_event["sample"].item()
    np.testing.assert_allclose(anomaly.values, expected_by_sample[sample])
    assert anomaly.attrs["month"] == selected_event["month"].item()
    assert anomaly.attrs["year"] == selected_event["year"].item()
    assert anomaly.attrs["sample"] == sample
    assert anomaly.attrs["file_name"] == selected_event["file_name"].item()
    assert anomaly.attrs["event"] == selected_event["event"].item()


def test_build_selected_event_composite_averages_selected_anomalies(
    tmp_path: Path,
) -> None:
    diagnostics = build_monthly_diagnostics(build_diffusion_dataset())
    diagnostics.attrs["source_input_dir"] = str(tmp_path)
    selection = build_enso_selection_dataset(diagnostics, months=[12])
    write_source_files(
        tmp_path,
        "month12_2030.nc",
        "month12_2031.nc",
        "month12_2032.nc",
    )

    selected_events = selection.where(selection["event_type"] == "nino", drop=True)
    composite = build_selected_event_composite(selected_events)

    sample_values = selected_events["sample"].values.tolist()
    expected_by_sample = {
        0: np.zeros((2, 2)),
        1: np.ones((2, 2)),
        2: -np.ones((2, 2)),
    }
    expected = np.mean([expected_by_sample[sample] for sample in sample_values], axis=0)

    np.testing.assert_allclose(composite.values, expected)
    assert composite.attrs["months"] == "December"
    assert composite.attrs["years"] == "2030,2031"
    assert composite.attrs["n_events"] == 2
    assert composite.attrs["event_types"] == "nino"


def test_build_enso_composites_dataset_returns_year_event_type_grid(
    tmp_path: Path,
) -> None:
    diagnostics = build_monthly_diagnostics(build_diffusion_dataset())
    diagnostics.attrs["source_input_dir"] = str(tmp_path)
    selection = build_enso_selection_dataset(diagnostics, months=[12])
    selection.attrs["source_dataset_path"] = str(tmp_path / "enso_selection.nc")
    write_source_files(
        tmp_path,
        "month12_2030.nc",
        "month12_2031.nc",
        "month12_2032.nc",
    )

    composites = build_enso_composites_dataset(
        selection,
        variables=["TREFHT"],
        years=[2030, 2031],
        num_workers=1,
    )

    assert composites["trefht_composite"].dims == ("event_type", "year", "lat", "lon")
    assert composites["trefht_composite"].shape == (2, 2, 2, 2)
    assert composites["n_events"].sel(event_type="nino", year=2030).item() == 1
    assert composites["n_events"].sel(event_type="nina", year=2030).item() == 1
    assert composites.attrs["pipeline_step"] == "enso_composites"
    assert composites.attrs["variables_processed"] == "TREFHT"


def test_get_diffusion_source_file_name_returns_expected_source_name() -> None:
    diffusion = build_diffusion_dataset()

    file_name = get_diffusion_source_file_name(diffusion, month=12, year=2031)

    assert file_name == "month12_2031.nc"


def test_get_diffusion_source_file_name_supports_numeric_lookup_on_labeled_months() -> (
    None
):
    diagnostics = build_monthly_diagnostics(build_diffusion_dataset())

    file_name = get_diffusion_source_file_name(diagnostics, month=12, year=2031)

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


def test_build_event_anomaly_matrix_returns_labeled_flattened_fields(
    tmp_path: Path,
) -> None:
    diffusion = build_diffusion_dataset()
    write_source_files(tmp_path, "month12_2030.nc", "month12_2031.nc")

    december = select_month(compute_nino34_index(diffusion), 12)
    events = december.stack(event=("year", "sample")).dropna("event").isel(event=[1, 3])

    matrix = build_event_anomaly_matrix(
        diffusion,
        events,
        input_data_dir=tmp_path,
        lat_bounds=(-20.0, 20.0),
        lon_bounds=(120.0, 280.0),
    )

    assert matrix.dims == ("event", "feature")
    assert matrix.shape == (2, 4)
    assert matrix.name == "trefht_anomaly_matrix"
    assert matrix["year"].values.tolist() == [2030, 2031]
    assert matrix["sample"].values.tolist() == [1, 0]
    assert matrix["month"].item() == 12
    assert matrix["lat"].values.tolist() == [-10.0, -10.0, 0.0, 0.0]
    assert matrix["lon"].values.tolist() == [190.0, 220.0, 190.0, 220.0]

    np.testing.assert_allclose(
        matrix.sel(event=events.event.values[0]).values, np.ones(4)
    )
    np.testing.assert_allclose(
        matrix.sel(event=events.event.values[1]).values, np.zeros(4)
    )

    first_field = matrix.sel(event=events.event.values[0]).unstack("feature")
    np.testing.assert_allclose(first_field.values, np.ones((2, 2)))


def test_build_event_anomaly_matrix_supports_multiple_workers(
    tmp_path: Path,
) -> None:
    diffusion = build_diffusion_dataset()
    write_source_files(tmp_path, "month12_2030.nc", "month12_2031.nc")

    december = select_month(compute_nino34_index(diffusion), 12)
    events = december.stack(event=("year", "sample")).dropna("event").isel(event=[1, 3])

    matrix = build_event_anomaly_matrix(
        diffusion,
        events,
        input_data_dir=tmp_path,
        lat_bounds=(-20.0, 20.0),
        lon_bounds=(120.0, 280.0),
        num_workers=2,
    )

    assert matrix.dims == ("event", "feature")
    assert matrix["year"].values.tolist() == [2030, 2031]
    assert matrix["sample"].values.tolist() == [1, 0]
    np.testing.assert_allclose(
        matrix.sel(event=events.event.values[0]).values, np.ones(4)
    )
    np.testing.assert_allclose(
        matrix.sel(event=events.event.values[1]).values, np.zeros(4)
    )


def test_build_event_anomaly_matrix_rejects_empty_event_selection(
    tmp_path: Path,
) -> None:
    diffusion = build_diffusion_dataset()
    events = xr.DataArray(
        np.array([], dtype=float),
        dims=("event",),
        coords={
            "event": np.array([], dtype=int),
            "year": ("event", np.array([], dtype=int)),
            "sample": ("event", np.array([], dtype=int)),
            "month": 12,
        },
    )

    try:
        build_event_anomaly_matrix(
            diffusion,
            events,
            input_data_dir=tmp_path,
        )
    except ValueError as exc:
        assert "at least one event" in str(exc)
    else:
        raise AssertionError(
            "expected build_event_anomaly_matrix to reject empty events"
        )
