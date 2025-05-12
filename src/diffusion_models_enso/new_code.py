import glob
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
import xskillscore as xs

from diffusion_models_enso.config import DataRetrievinSettings
from diffusion_models_enso.utils.load_config import get_data_retrieving_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_settings = get_data_retrieving_settings()


@dataclass
class ClimateDataResult:
    ensemble_means: Dict[int, np.ndarray]
    ensemble_stds: Dict[int, np.ndarray]
    lens_means: Dict[int, np.ndarray]
    lens_stds: Dict[int, np.ndarray]
    ensemble_file_paths: Dict[int, str]
    lens_file_paths: Dict[int, List[str]]
    lens_closest_indexes: Dict[int, List[int]]


def compute_weighted_mean_std(
    da: xr.DataArray, weights: xr.DataArray, resample_dim: str, resample_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    mean = (
        da.weighted(weights).mean(dim=[d for d in da.dims if d != resample_dim]).values
    )
    DDS = xs.resample_iterations_idx(da, resample_size, resample_dim)
    std = (
        DDS.weighted(weights)
        .mean(dim=[resample_dim] + [d for d in da.dims if d != resample_dim])
        .std("iteration")
    )
    return mean, std.values


def extract_co2vmr_from_filename(fn: str) -> float:
    return float(fn.split("co2")[-1].split("_")[0])


def closest_co2vmr_year(co2vmr: float, file_paths: List[str]) -> Tuple[int, int, str]:
    closest_diff = np.inf
    closest_time = None
    closest_file = None

    for fn in file_paths:
        ds = xr.open_dataset(fn)
        times = ds["time"].values
        co2_values = ds["co2vmr"].values

        abs_diff = np.abs(co2_values - co2vmr)
        min_index = np.argmin(abs_diff)

        if abs_diff[min_index] < closest_diff:
            closest_diff = abs_diff[min_index]
            closest_time = times[min_index]
            closest_file = fn

    if closest_time is None or closest_file is None:
        raise ValueError("No suitable CO₂ VMR match found in the provided files.")

    logger.info(f"Closest match found in file: {closest_file}, diff: {closest_diff}")
    return int(closest_time.year), int(closest_time.month), closest_file


def extract_closest_co2vmr_data(
    co2vmr: float, target_month: int, variable: str, file_paths: List[str]
) -> Tuple[xr.Dataset, List[int]]:
    dats_list = []
    closest_indexes = []

    for fn in file_paths:
        ds = xr.open_dataset(fn)
        ds_month = ds.where(ds["time.month"] == target_month, drop=True)

        co2_values = ds_month["co2vmr"].values
        abs_diff = np.abs(co2_values - co2vmr)
        min_index = np.argmin(abs_diff)

        closest_indexes.append(min_index)
        selected_data = ds_month.isel(time=min_index)[variable].values
        dats_list.append(selected_data)

    dats_stack = np.stack(dats_list, axis=0)
    lat = ds_month["lat"].values
    lon = ds_month["lon"].values
    times = np.arange(len(file_paths))

    dats_da = xr.DataArray(
        dats_stack,
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lat, "lon": lon},
    )

    logger.info(f"Extracted data shape: {dats_stack.shape} for variable: {variable}")
    return dats_da.to_dataset(name=variable), closest_indexes


def get_matching_lens_data(
    co2vmr: float,
    target_month: int,
    variable: str,
    settings: DataRetrievinSettings = base_settings,
) -> Tuple[xr.Dataset, List[int]]:
    lens_files = sorted(glob.glob(f"{settings.lens_data_path}/*1001.001*.nc"))
    yt, mt, fout = closest_co2vmr_year(co2vmr, lens_files)

    if mt != ((target_month + 1) % 12):
        logger.warning("Mismatched target month vs LENS data month")

    year = int(fout.split(".")[-2][:4])
    year_files = sorted(glob.glob(f"{settings.lens_data_path}/*.{year}*.nc"))
    return extract_closest_co2vmr_data(co2vmr, mt, variable, year_files)


def get_data(
    momo: int, flag: bool = False, settings: DataRetrievinSettings = base_settings
) -> ClimateDataResult:

    pattern = f"{settings.diffusion_data_path}{momo:02}*.nc"
    ensemble_files = sorted(glob.glob(pattern))

    result = ClimateDataResult(
        ensemble_means={},
        ensemble_stds={},
        lens_means={},
        lens_stds={},
        ensemble_file_paths={},
        lens_file_paths={},
        lens_closest_indexes={},
    )

    for ee, fn in enumerate(ensemble_files):
        logger.info(f"Processing ensemble {ee + 2015}: {fn}")
        ds = xr.open_dataset(fn)

        if flag:
            ds = ds.sel(
                lat=slice(settings.ENSO_lat_min, settings.ENSO_lat_max),
                lon=slice(settings.ENSO_long_min, settings.ENSO_long_max),
            )

        lat = ds["lat"]
        weights = np.cos(np.deg2rad(lat))
        mean, std = compute_weighted_mean_std(ds["TREFHT"], weights, "samples", 50)

        result.ensemble_means[ee] = mean
        result.ensemble_stds[ee] = std
        result.ensemble_file_paths[ee] = fn

        co2vmr = extract_co2vmr_from_filename(fn)
        lens_ds, closest_indexes = get_matching_lens_data(
            co2vmr, momo, "TREFHT", settings
        )

        result.lens_file_paths[ee] = lens_ds.attrs.get("source_files", [])
        result.lens_closest_indexes[ee] = closest_indexes

        if flag:
            lens_ds = lens_ds.sel(
                lat=slice(settings.ENSO_lat_min, settings.ENSO_lat_max),
                lon=slice(settings.ENSO_long_min, settings.ENSO_long_max),
            )

        mean_lens, std_lens = compute_weighted_mean_std(
            lens_ds["TREFHT"], weights, "time", 50
        )
        result.lens_means[ee] = mean_lens
        result.lens_stds[ee] = std_lens

    logger.info("Data extraction and processing completed.")
    return result
