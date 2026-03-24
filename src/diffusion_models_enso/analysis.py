from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr
from tqdm import tqdm

from diffusion_models_enso.utils import drop_empty_years

LOGGER = logging.getLogger(__name__)

DEFAULT_MONTHS = (12, 1, 2, 6, 7, 8)
DEFAULT_ENSO_MONTHS = (12, 1, 2)
DEFAULT_COMPOSITE_YEARS = (2015, 2025, 2035, 2045, 2055, 2065, 2075, 2085)
DEFAULT_SOURCE_FILE_PATTERN = "samples_governance_indexes_3944_month{month:02}*.nc"
DEFAULT_SOURCE_VARIABLE = "TREFHT"
DEFAULT_NINO34_VARIABLE = "nino34_trefht"
DEFAULT_QUANTILE_NINO = 0.85
DEFAULT_QUANTILE_NINA = 0.15
DEFAULT_COMPOSITE_VARIABLES = ("TREFHT", "PS", "PRECT")

MONTH_NUMBER_TO_NAME = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}
MONTH_NAME_TO_NUMBER = {
    month_name: month_number
    for month_number, month_name in MONTH_NUMBER_TO_NAME.items()
}


def month_number_to_name(month: int) -> str:
    """Return the canonical month label for a numeric month."""
    try:
        return MONTH_NUMBER_TO_NAME[month]
    except KeyError as exc:
        raise ValueError(f"Unsupported month number: {month}") from exc


def _build_dataset_id(pipeline_step: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{pipeline_step}:{timestamp}"


def _get_source_input_dir(data: xr.Dataset | xr.DataArray) -> Path:
    source_input_dir = data.attrs.get("source_input_dir", "")
    if not source_input_dir:
        raise ValueError(
            "Dataset is missing the 'source_input_dir' attribute needed to locate raw files"
        )
    return Path(source_input_dir)


def _get_month_numbers(
    data: xr.Dataset | xr.DataArray,
    *,
    month_dim: str,
    month_number_name: str,
) -> np.ndarray:
    if month_number_name in data.coords:
        month_numbers = np.asarray(data.coords[month_number_name].values)
    else:
        month_numbers = np.asarray(data.coords[month_dim].values)

    try:
        return month_numbers.astype(int)
    except ValueError as exc:
        raise ValueError(
            f"Could not derive numeric month values from '{month_dim}' or '{month_number_name}'"
        ) from exc


def _resolve_month_value(
    data: xr.Dataset | xr.DataArray,
    month: int | str,
    *,
    month_dim: str,
    month_number_name: str,
) -> str | int:
    if isinstance(month, str):
        return month

    if month_number_name in data.coords:
        month_numbers = data.coords[month_number_name]
        matches = data.coords[month_dim].where(month_numbers == month, drop=True)
        if matches.size == 0:
            raise KeyError(f"Month {month} not found in '{month_number_name}'")
        return matches.item()

    return month


def _validate_same_grid(
    source_dataset: xr.Dataset,
    reference_lat: np.ndarray,
    reference_lon: np.ndarray,
    *,
    file_name: str,
    lat_dim: str,
    lon_dim: str,
) -> None:
    if not np.array_equal(source_dataset[lat_dim].to_numpy(), reference_lat):
        raise ValueError(f"File {file_name} has a different latitude grid")
    if not np.array_equal(source_dataset[lon_dim].to_numpy(), reference_lon):
        raise ValueError(f"File {file_name} has a different longitude grid")


def build_month_arrays(
    files: list[Path],
    month: int,
    n_samples: int | None,
    *,
    variable: str = DEFAULT_SOURCE_VARIABLE,
    source_sample_dim: str = "samples",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> tuple[np.ndarray, np.ndarray, list[str], int, int]:
    """Compute global and Nino3.4 mean TREFHT arrays for one month."""
    if not files:
        raise FileNotFoundError(f"No files found for month {month:02}")

    with xr.open_dataset(files[0]) as first_ds:
        month_n_samples = first_ds.sizes[source_sample_dim]
        reference_lat = first_ds[lat_dim].to_numpy()
        reference_lon = first_ds[lon_dim].to_numpy()

    if n_samples is not None and month_n_samples != n_samples:
        raise ValueError(
            f"Month {month:02} has {month_n_samples} samples, expected {n_samples}"
        )

    month_n_years = len(files)
    global_temp = np.empty((month_n_years, month_n_samples))
    nino34_trefht = np.empty((month_n_years, month_n_samples))

    for index_year, file_path in enumerate(
        tqdm(files, desc=f"Processing month {month:02}", leave=False)
    ):
        with xr.open_dataset(file_path) as source_dataset:
            if source_dataset.sizes[source_sample_dim] != month_n_samples:
                raise ValueError(
                    f"File {file_path.name} has {source_dataset.sizes[source_sample_dim]} samples, "
                    f"expected {month_n_samples}"
                )

            _validate_same_grid(
                source_dataset,
                reference_lat,
                reference_lon,
                file_name=file_path.name,
                lat_dim=lat_dim,
                lon_dim=lon_dim,
            )

            weights = xr.DataArray(
                np.cos(np.deg2rad(source_dataset[lat_dim].to_numpy())),
                coords={lat_dim: source_dataset[lat_dim]},
                dims=(lat_dim,),
            )
            source_field = source_dataset[variable]
            global_samples = source_field.weighted(weights=weights).mean(
                dim=(lat_dim, lon_dim)
            )

            nino_region = source_field.sel(
                {lat_dim: slice(-5, 5), lon_dim: slice(190, 240)}
            )
            nino_weights = xr.DataArray(
                np.cos(np.deg2rad(nino_region[lat_dim].to_numpy())),
                coords={lat_dim: nino_region[lat_dim]},
                dims=(lat_dim,),
            )
            nino_samples = nino_region.weighted(weights=nino_weights).mean(
                dim=(lat_dim, lon_dim)
            )

            global_temp[index_year, :] = global_samples.values
            nino34_trefht[index_year, :] = nino_samples.values

    return (
        global_temp,
        nino34_trefht,
        [file_path.name for file_path in files],
        month_n_years,
        month_n_samples,
    )


def build_monthly_diagnostics(
    month_index_dataset: xr.Dataset,
    *,
    nino34_variable: str = DEFAULT_NINO34_VARIABLE,
    nino34_index_name: str = "nino34_index",
    sample_dim: str = "sample",
    month_dim: str = "month",
    month_number_name: str = "month_number",
) -> xr.Dataset:
    """Build a monthly diagnostics dataset from per-month aggregate inputs."""
    diagnostics = month_index_dataset.copy()
    if nino34_variable not in diagnostics.data_vars:
        raise KeyError(
            f"Dataset does not contain the required variable '{nino34_variable}'"
        )
    if nino34_variable != DEFAULT_NINO34_VARIABLE:
        diagnostics = diagnostics.rename({nino34_variable: DEFAULT_NINO34_VARIABLE})

    month_numbers = _get_month_numbers(
        diagnostics,
        month_dim=month_dim,
        month_number_name=month_number_name,
    )
    month_labels = np.asarray(
        [month_number_to_name(month_number) for month_number in month_numbers]
    )

    diagnostics = diagnostics.assign_coords(
        {
            month_dim: (month_dim, month_labels),
            month_number_name: (month_dim, month_numbers),
        }
    )
    diagnostics[nino34_index_name] = compute_nino34_index(
        diagnostics,
        variable=DEFAULT_NINO34_VARIABLE,
        sample_dim=sample_dim,
    )
    diagnostics.attrs = {
        **month_index_dataset.attrs,
        "dataset_id": month_index_dataset.attrs.get(
            "dataset_id",
            _build_dataset_id("monthly_diagnostics"),
        ),
        "pipeline_step": "monthly_diagnostics",
        "month_coordinate": month_dim,
        "month_number_coordinate": month_number_name,
        "nino34_index_definition": (
            "nino34_trefht minus ensemble mean across sample for each month/year group"
        ),
    }
    return diagnostics


def build_monthly_diagnostics_dataset(
    input_dir: Path,
    months: Sequence[int] = DEFAULT_MONTHS,
    start_year: int = 2015,
    *,
    variable: str = DEFAULT_SOURCE_VARIABLE,
    file_pattern: str = DEFAULT_SOURCE_FILE_PATTERN,
) -> xr.Dataset:
    """Build the first persisted monthly diagnostics artifact from raw monthly files."""
    selected_months = tuple(months)
    n_samples: int | None = None
    global_by_month: list[np.ndarray] = []
    nino34_by_month: list[np.ndarray] = []
    file_names_by_month: list[list[str]] = []
    year_counts: list[int] = []

    for month in selected_months:
        files = sorted(Path(input_dir).glob(file_pattern.format(month=month)))
        global_temp, nino34_trefht, file_names, month_n_years, month_n_samples = (
            build_month_arrays(
                files,
                month,
                n_samples,
                variable=variable,
            )
        )

        if n_samples is None:
            n_samples = month_n_samples

        global_by_month.append(global_temp)
        nino34_by_month.append(nino34_trefht)
        file_names_by_month.append(file_names)
        year_counts.append(month_n_years)

    if n_samples is None:
        raise RuntimeError("No data was processed")

    n_years = max(year_counts)
    years = [start_year + index for index in range(n_years)]
    global_array = np.full((len(selected_months), n_years, n_samples), np.nan)
    nino34_array = np.full((len(selected_months), n_years, n_samples), np.nan)
    max_name_len = max(len(name) for names in file_names_by_month for name in names)
    file_name_array = np.full(
        (len(selected_months), n_years),
        "",
        dtype=f"<U{max_name_len}",
    )

    for month_index, month_n_years in enumerate(year_counts):
        global_array[month_index, :month_n_years, :] = global_by_month[month_index]
        nino34_array[month_index, :month_n_years, :] = nino34_by_month[month_index]
        file_name_array[month_index, :month_n_years] = file_names_by_month[month_index]

    month_index_dataset = xr.Dataset(
        data_vars={
            "global_trefht": (("month", "year", "sample"), global_array),
            DEFAULT_NINO34_VARIABLE: (("month", "year", "sample"), nino34_array),
            "file_name": (("month", "year"), file_name_array),
        },
        coords={
            "month": np.asarray(selected_months),
            "year": years,
            "sample": np.arange(n_samples),
        },
        attrs={
            "dataset_id": _build_dataset_id("monthly_diagnostics"),
            "description": "Monthly diagnostics derived from diffusion NetCDF files",
            "source_input_dir": str(Path(input_dir)),
            "source_variable": variable,
            "source_file_pattern": file_pattern,
            "months_processed": ",".join(str(month) for month in selected_months),
            "start_year": start_year,
            "global_weighting": "cos(lat)",
            "nino34_weighting": "cos(lat)",
            "nino34_lat_range": "-5 to 5",
            "nino34_lon_range": "190 to 240",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return build_monthly_diagnostics(month_index_dataset)


def compute_nino34_index(
    diffusion_dataset: xr.Dataset,
    *,
    variable: str = DEFAULT_NINO34_VARIABLE,
    sample_dim: str = "sample",
) -> xr.DataArray:
    """Center the stored Nino 3.4 temperatures by subtracting the ensemble mean."""
    nino34 = diffusion_dataset[variable]
    return nino34 - nino34.mean(dim=sample_dim, skipna=True)


def select_month(
    data_array: xr.DataArray,
    month: int | str,
    *,
    month_dim: str = "month",
    month_number_name: str = "month_number",
    sample_dim: str = "sample",
    year_dim: str = "year",
    drop_empty: bool = True,
) -> xr.DataArray:
    """Select one month and optionally drop year rows with no valid samples."""
    month_value = _resolve_month_value(
        data_array,
        month,
        month_dim=month_dim,
        month_number_name=month_number_name,
    )
    month_data = data_array.sel({month_dim: month_value})
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
    neutral_sample = np.abs(selected).idxmin(sample_dim)  # type: ignore

    return (
        selected.sel({sample_dim: max_sample}),
        selected.sel({sample_dim: min_sample}),
        selected.sel({sample_dim: neutral_sample}),
    )


def build_enso_selection_dataset(
    monthly_diagnostics: xr.Dataset,
    *,
    months: Sequence[int | str] = DEFAULT_ENSO_MONTHS,
    quantile_nino: float = DEFAULT_QUANTILE_NINO,
    quantile_nina: float = DEFAULT_QUANTILE_NINA,
    index_variable: str = "nino34_index",
    file_name_variable: str = "file_name",
    month_dim: str = "month",
    month_number_name: str = "month_number",
    source_dataset_path: str | None = None,
) -> xr.Dataset:
    """Build a persisted ENSO event-selection dataset from monthly diagnostics."""
    if index_variable not in monthly_diagnostics.data_vars:
        raise KeyError(
            f"Dataset does not contain the required variable '{index_variable}'"
        )
    if file_name_variable not in monthly_diagnostics.data_vars:
        raise KeyError(
            f"Dataset does not contain the required variable '{file_name_variable}'"
        )

    selected_months = [
        _resolve_month_value(
            monthly_diagnostics,
            month,
            month_dim=month_dim,
            month_number_name=month_number_name,
        )
        for month in months
    ]

    index_data = monthly_diagnostics[index_variable].sel({month_dim: selected_months})
    file_name_data = (
        monthly_diagnostics[file_name_variable]
        .sel({month_dim: selected_months})
        .broadcast_like(index_data)
    )
    month_number_data = (
        monthly_diagnostics[month_number_name]
        .sel({month_dim: selected_months})
        .broadcast_like(index_data)
    )

    selection_data = xr.Dataset(
        data_vars={
            index_variable: index_data,
            file_name_variable: file_name_data,
            month_number_name: month_number_data,
        }
    )
    selection_data = selection_data.stack(event=(month_dim, "year", "sample"))
    selection_data = selection_data.dropna(dim="event", subset=[index_variable])
    selection_data = selection_data.reset_index("event")
    selection_data = selection_data.assign_coords(
        event=("event", np.arange(selection_data.sizes["event"]))
    )

    if selection_data.sizes["event"] == 0:
        raise ValueError("No valid nino34_index values found for the selected months")

    index_values = selection_data[index_variable]
    threshold_nino = index_values.quantile(quantile_nino, dim="event").item()
    threshold_nina = index_values.quantile(quantile_nina, dim="event").item()

    is_selected = (index_values > threshold_nino) | (index_values < threshold_nina)
    selection_data = selection_data.where(is_selected, drop=True)

    if selection_data.sizes["event"] == 0:
        raise ValueError("No ENSO events matched the selected thresholds")

    event_type = np.where(
        selection_data[index_variable].values > threshold_nino,
        "nino",
        "nina",
    )
    selection_data["event_type"] = xr.DataArray(
        event_type,
        dims=("event",),
        coords={"event": selection_data["event"]},
    )
    selection_data.attrs = {
        "dataset_id": _build_dataset_id("enso_selection"),
        "pipeline_step": "enso_selection",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset_path": source_dataset_path or "",
        "source_dataset_id": monthly_diagnostics.attrs.get("dataset_id", ""),
        "source_input_dir": monthly_diagnostics.attrs.get("source_input_dir", ""),
        "source_index_variable": index_variable,
        "selection_months": ",".join(str(month) for month in selected_months),
        "selection_scope": "shared_threshold_across_months",
        "quantile_nino": quantile_nino,
        "quantile_nina": quantile_nina,
        "threshold_nino": threshold_nino,
        "threshold_nina": threshold_nina,
    }
    return selection_data


def get_diffusion_source_file_name(
    diffusion_dataset: xr.Dataset,
    month: int | str,
    year: int,
    *,
    variable: str = "file_name",
    month_dim: str = "month",
    month_number_name: str = "month_number",
    year_dim: str = "year",
) -> str:
    """Look up the original monthly NetCDF file recorded for a month/year pair."""
    month_value = _resolve_month_value(
        diffusion_dataset,
        month,
        month_dim=month_dim,
        month_number_name=month_number_name,
    )
    return (
        diffusion_dataset[variable].sel({month_dim: month_value, year_dim: year}).item()
    )


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


def prepare_selected_event_anomaly(
    selected_event: xr.Dataset,
    *,
    variable: str = "TREFHT",
    file_name_var: str = "file_name",
    sample_var: str = "sample",
    source_sample_dim: str = "samples",
) -> xr.DataArray:
    """Rebuild an anomaly field from one selected ENSO event."""
    file_name = selected_event[file_name_var].item()
    sample = selected_event[sample_var].item()
    input_data_dir = _get_source_input_dir(selected_event)
    source_path = Path(input_data_dir) / file_name
    with xr.open_dataset(source_path) as source_dataset:
        anomaly = compute_sample_anomaly(
            source_dataset,
            sample=sample,
            variable=variable,
            sample_dim=source_sample_dim,
        ).load()
    anomaly.attrs = {
        **anomaly.attrs,
        "month": selected_event["month"].item(),
        "year": int(selected_event["year"].item()),
        "sample": int(sample),
        "file_name": file_name,
    }
    if "event" in selected_event.coords:
        anomaly.attrs["event"] = int(selected_event["event"].item())
    return anomaly


def reconstruct_selected_event_anomaly(
    selection_dataset: xr.Dataset,
    event: int,
    *,
    variable: str = "TREFHT",
    file_name_var: str = "file_name",
    sample_var: str = "sample",
    source_sample_dim: str = "samples",
) -> xr.DataArray:
    """Rebuild a selected event anomaly field directly from the ENSO selection dataset."""
    selected_event = selection_dataset.sel(event=event)
    return prepare_selected_event_anomaly(
        selected_event,
        variable=variable,
        file_name_var=file_name_var,
        sample_var=sample_var,
        source_sample_dim=source_sample_dim,
    )


def build_selected_event_composite(
    selected_events: xr.Dataset,
    *,
    variable: str = "TREFHT",
    file_name_var: str = "file_name",
    sample_var: str = "sample",
    source_sample_dim: str = "samples",
) -> xr.DataArray:
    """Rebuild and average anomaly fields for a selected event subset."""
    if "event" not in selected_events.dims:
        raise ValueError("selected_events must have an 'event' dimension")
    if selected_events.sizes["event"] == 0:
        raise ValueError("selected_events must contain at least one event")

    anomalies = [
        prepare_selected_event_anomaly(
            selected_events.sel(event=event_value),
            variable=variable,
            file_name_var=file_name_var,
            sample_var=sample_var,
            source_sample_dim=source_sample_dim,
        )
        for event_value in selected_events["event"].values
    ]
    composite = xr.concat(anomalies, dim="event").mean(dim="event", skipna=True)

    months = (
        [selected_events["month"].item()]
        if selected_events["month"].ndim == 0
        else selected_events["month"].values.tolist()
    )
    years = (
        [int(selected_events["year"].item())]
        if selected_events["year"].ndim == 0
        else [int(year) for year in selected_events["year"].values.tolist()]
    )
    composite.attrs = {
        **composite.attrs,
        "months": ",".join(dict.fromkeys(months)),
        "years": ",".join(str(year) for year in dict.fromkeys(years)),
        "n_events": int(selected_events.sizes["event"]),
        "variable": variable,
    }
    if "event_type" in selected_events:
        event_types = selected_events["event_type"].values.tolist()
        composite.attrs["event_types"] = ",".join(dict.fromkeys(event_types))
    return composite


def _build_variable_composites(
    selection_dataset: xr.Dataset,
    *,
    variable: str,
    years: Sequence[int],
    event_types: Sequence[str],
) -> tuple[str, xr.DataArray, xr.DataArray]:
    # Use one reconstructed anomaly as the template grid for the whole
    # event_type x year composite cube of this variable.
    reference_event = selection_dataset.isel(event=0)
    reference_composite = prepare_selected_event_anomaly(
        reference_event,
        variable=variable,
    )
    lat_values = reference_composite["lat"].values
    lon_values = reference_composite["lon"].values

    composite_shape = (len(event_types), len(years), len(lat_values), len(lon_values))
    composite_values = np.full(composite_shape, np.nan, dtype=float)
    n_events_values = np.zeros((len(event_types), len(years)), dtype=int)

    for event_type_index, event_type in enumerate(event_types):
        for year_index, year in enumerate(years):
            # Each composite is defined by all selected events of one event_type
            # in one target year.
            selected_events = selection_dataset.where(
                (selection_dataset["event_type"] == event_type)
                & (selection_dataset["year"] == year),
                drop=True,
            )
            if selected_events.sizes["event"] == 0:
                continue

            composite = build_selected_event_composite(
                selected_events,
                variable=variable,
            )
            composite_values[event_type_index, year_index, :, :] = composite.values
            n_events_values[event_type_index, year_index] = selected_events.sizes["event"]

    composite_name = f"{variable.lower()}_composite"
    composite_array = xr.DataArray(
        composite_values,
        dims=("event_type", "year", "lat", "lon"),
        coords={
            "event_type": list(event_types),
            "year": list(years),
            "lat": lat_values,
            "lon": lon_values,
        },
        name=composite_name,
        attrs={"source_variable": variable},
    )
    n_events = xr.DataArray(
        n_events_values,
        dims=("event_type", "year"),
        coords={
            "event_type": list(event_types),
            "year": list(years),
        },
        name=f"{variable.lower()}_n_events",
        attrs={"source_variable": variable},
    )
    return composite_name, composite_array, n_events


def _build_variable_composites_from_path(
    selection_path: Path,
    *,
    variable: str,
    years: Sequence[int],
    event_types: Sequence[str],
) -> tuple[str, xr.DataArray, xr.DataArray]:
    with xr.open_dataset(selection_path) as selection_dataset:
        return _build_variable_composites(
            selection_dataset.load(),
            variable=variable,
            years=years,
            event_types=event_types,
        )


def _build_variable_composites_from_job(
    job: tuple[Path, str, tuple[int, ...], tuple[str, ...]],
) -> tuple[str, xr.DataArray, xr.DataArray]:
    selection_path, variable, years, event_types = job
    return _build_variable_composites_from_path(
        selection_path,
        variable=variable,
        years=years,
        event_types=event_types,
    )


def build_enso_composites_dataset(
    selection_dataset: xr.Dataset,
    *,
    variables: Sequence[str] = DEFAULT_COMPOSITE_VARIABLES,
    years: Sequence[int] = DEFAULT_COMPOSITE_YEARS,
    event_types: Sequence[str] = ("nino", "nina"),
    source_dataset_path: str | None = None,
    num_workers: int | None = None,
) -> xr.Dataset:
    """Build year-by-year ENSO composites for one or more source variables."""
    if "event" not in selection_dataset.dims or selection_dataset.sizes["event"] == 0:
        raise ValueError("selection_dataset must contain at least one event")

    unique_variables = list(dict.fromkeys(variables))
    unique_years = [int(year) for year in dict.fromkeys(years)]
    unique_event_types = list(dict.fromkeys(event_types))

    if num_workers is None:
        num_workers = min(len(unique_variables), os.cpu_count() or 1)
    else:
        num_workers = max(1, min(num_workers, len(unique_variables)))

    variable_results: list[tuple[str, xr.DataArray, xr.DataArray]]
    if num_workers == 1:
        # The serial path is the simplest execution mode and is also used when
        # only one variable was requested.
        variable_results = [
            _build_variable_composites(
                selection_dataset,
                variable=variable,
                years=unique_years,
                event_types=unique_event_types,
            )
            for variable in tqdm(unique_variables, desc="Building composites")
        ]
    else:
        selection_path = source_dataset_path or selection_dataset.attrs.get(
            "source_dataset_path", ""
        )
        if not selection_path:
            raise ValueError(
                "Parallel composite building requires 'source_dataset_path' so workers can reopen the selection dataset"
            )
        # Keep parallelism deliberately simple: one worker handles one source
        # variable and builds all year/event_type composites for it.
        jobs = [
            (Path(selection_path), variable, tuple(unique_years), tuple(unique_event_types))
            for variable in unique_variables
        ]
        future_to_variable: dict[object, str] = {}
        results_by_variable: dict[str, tuple[str, xr.DataArray, xr.DataArray]] = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for job in jobs:
                future = executor.submit(_build_variable_composites_from_job, job)
                future_to_variable[future] = job[1]

            for future in tqdm(
                as_completed(future_to_variable),
                total=len(future_to_variable),
                desc="Building composites",
            ):
                variable_name = future_to_variable[future]
                LOGGER.info("Finished composite build for %s", variable_name)
                results_by_variable[variable_name] = future.result()

        variable_results = [
            results_by_variable[variable]
            for variable in unique_variables
        ]

    composite_dataset = xr.Dataset(
        data_vars={
            result_name: result_array
            for result_name, result_array, _ in variable_results
        }
    )
    # Event counts are shared across variables because the ENSO selection is the
    # same; only the reconstructed source variable changes.
    composite_dataset["n_events"] = xr.DataArray(
        variable_results[0][2].values,
        dims=("event_type", "year"),
        coords={
            "event_type": variable_results[0][2]["event_type"].values,
            "year": variable_results[0][2]["year"].values,
        },
    )
    composite_dataset.attrs = {
        "dataset_id": _build_dataset_id("enso_composites"),
        "pipeline_step": "enso_composites",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset_path": source_dataset_path
        or selection_dataset.attrs.get("source_dataset_path", ""),
        "source_dataset_id": selection_dataset.attrs.get("dataset_id", ""),
        "source_input_dir": selection_dataset.attrs.get("source_input_dir", ""),
        "selection_months": selection_dataset.attrs.get("selection_months", ""),
        "variables_processed": ",".join(unique_variables),
        "years_processed": ",".join(str(year) for year in unique_years),
        "event_types": ",".join(unique_event_types),
        "quantile_nino": selection_dataset.attrs.get("quantile_nino", ""),
        "quantile_nina": selection_dataset.attrs.get("quantile_nina", ""),
        "threshold_nino": selection_dataset.attrs.get("threshold_nino", ""),
        "threshold_nina": selection_dataset.attrs.get("threshold_nina", ""),
    }
    return composite_dataset


def _build_event_anomaly_row(
    source_path: Path,
    sample: int,
    variable: str,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
) -> xr.DataArray:
    with xr.open_dataset(source_path) as source_dataset:
        anomaly = compute_sample_anomaly(
            source_dataset,
            sample=sample,
            variable=variable,
        ).load()

    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds
    cropped = anomaly.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
    )
    return cropped.stack(feature=("lat", "lon"))


def _build_event_anomaly_chunk(
    worker_index: int,
    jobs: list[tuple[Path, int, str, tuple[float, float], tuple[float, float]]],
) -> list[xr.DataArray]:
    rows: list[xr.DataArray] = []
    for job in tqdm(
        jobs,
        position=worker_index,
        desc=f"worker {worker_index + 1}",
    ):
        rows.append(_build_event_anomaly_row(*job))
    return rows


def build_event_anomaly_matrix(
    diffusion_dataset: xr.Dataset,
    events: xr.DataArray,
    input_data_dir: Path,
    *,
    variable: str = "TREFHT",
    lat_bounds: tuple[float, float] = (-20.0, 20.0),
    lon_bounds: tuple[float, float] = (120.0, 280.0),
    num_workers: int | None = None,
) -> xr.DataArray:
    """Reconstruct, crop, and flatten anomaly fields for an event selection."""
    if "event" not in events.dims:
        raise ValueError("events must have an 'event' dimension")
    if events.sizes["event"] == 0:
        raise ValueError("events must contain at least one event")

    jobs: list[tuple[Path, int, str, tuple[float, float], tuple[float, float]]] = []
    for event_value in events.event.values:
        selected_event = events.sel(event=event_value)
        file_name = get_diffusion_source_file_name(
            diffusion_dataset,
            month=selected_event["month"].item(),
            year=selected_event["year"].item(),
        )
        source_path = Path(input_data_dir) / file_name
        jobs.append(
            (
                source_path,
                selected_event["sample"].item(),
                variable,
                lat_bounds,
                lon_bounds,
            )
        )

    if num_workers is None:
        num_workers = os.cpu_count() or 1

    if num_workers == 1:
        rows = [_build_event_anomaly_row(*job) for job in tqdm(jobs, total=len(jobs))]
    else:
        num_workers = min(num_workers, len(jobs))
        chunk_size = (len(jobs) + num_workers - 1) // num_workers
        job_chunks = [
            jobs[index : index + chunk_size]
            for index in range(0, len(jobs), chunk_size)
        ]

        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                chunk_rows = list(
                    executor.map(
                        _build_event_anomaly_chunk,
                        range(len(job_chunks)),
                        job_chunks,
                    )
                )
            rows = [row for chunk in chunk_rows for row in chunk]
        except (NotImplementedError, PermissionError, OSError):
            rows = [
                _build_event_anomaly_row(*job) for job in tqdm(jobs, total=len(jobs))
            ]

    matrix = xr.concat(rows, dim="event")
    matrix = matrix.transpose("event", "feature")
    matrix.name = f"{variable.lower()}_anomaly_matrix"
    matrix = matrix.assign_coords(
        event=("event", np.asarray(events["event"].values, dtype=object))
    )
    for coord_name in ("year", "sample"):
        if coord_name in events.coords:
            matrix = matrix.assign_coords(
                {coord_name: ("event", np.asarray(events[coord_name].values))}
            )
    if "month" in events.coords and events["month"].ndim == 0:
        matrix = matrix.assign_coords(month=events["month"])
    elif "month" in events.coords:
        matrix = matrix.assign_coords(month=events["month"])

    matrix.attrs.update(
        {
            "variable": variable,
            "lat_bounds": lat_bounds,
            "lon_bounds": lon_bounds,
        }
    )
    return matrix
